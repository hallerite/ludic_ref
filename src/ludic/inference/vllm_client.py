import atexit
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests
import torch  # type: ignore
from openai import AsyncOpenAI
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from ludic.types import Message, ChatResponse
from ludic.inference.client import ChatClient
from ludic.inference.sampling import SamplingConfig

log = logging.getLogger(__name__)


class VLLMChatClient(ChatClient):
    """
    vLLM ChatClient backed by:
      - the OpenAI-compatible inference server
      - an optional NCCL-based weight update path.

    Modes:
      * inference-only (enable_weight_updates=False):
            - Only uses HTTP OpenAI API.
            - No NCCL, no GPU expected on client side.
            - Weight updates are disabled.
      * training/update mode (enable_weight_updates=True):
            - Client becomes an additional NCCL rank.
            - Enables sync_weights() to broadcast updated parameters
              directly into the vLLM worker processes.

    Args:
        host:
            Hostname of the vLLM OpenAI-compatible server. Defaults to "0.0.0.0".
        port:
            HTTP port for the vLLM server. Defaults to 8000.
        group_port:
            TCP port used to form the StatelessProcessGroup for NCCL-based
            weight updates. Only used when enable_weight_updates=True.
        connection_timeout_s:
            Maximum number of seconds to wait for the server /health endpoint
            to become reachable during initialization. Defaults to 60 seconds.
            If the timeout is exceeded, the constructor raises ConnectionError.
        enable_weight_updates:
            If True, initialize the NCCL communicator and enable
            sync_weights(); otherwise run in inference-only mode.
        device:
            The device (e.g. "cuda:0", 0, or torch.device) to bind the NCCL
            communicator to. Defaults to 0. Important when running client on
            multi-GPU setups (e.g. via accelerate).
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        group_port: int = 51216,
        connection_timeout_s: float = 60,
        enable_weight_updates: bool = False,
        device: Union[str, torch.device, int] = 0,
    ) -> None:

        # Store configuration parameters
        self.host = host
        self.port = port
        self.group_port = group_port
        self.connection_timeout_s = connection_timeout_s
        self.enable_weight_updates = enable_weight_updates
        self.device = device

        # AsyncOpenAI handles the OpenAI-compatible HTTP endpoints.
        self._async_client = AsyncOpenAI(
            base_url=f"http://{self.host}:{self.port}/v1",
            api_key="local",
        )

        # Sync HTTP client for health checks and weight-update metadata RPC
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self.server_url = f"http://{self.host}:{self.port}"
        self._pynccl_comm: Optional[PyNcclCommunicator] = None
        self._rank: Optional[int] = None

        # Verify server is reachable before continuing.
        self._check_server(self.connection_timeout_s)

        # If weight updates are enabled, the client forms the extra NCCL rank.
        if self.enable_weight_updates:
            self._init_communicator()
            atexit.register(self.close_communicator)

    # ---- ChatClient.complete ------------------------------------

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
        interrupt_thinking: Optional[int] = None,
        return_token_ids: bool = False,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        """
        High-level LLM invocation with vLLM extensions.

        Args:
            interrupt_thinking:
                If set to an integer N, injects:
                    extra_body["vllm_xargs"]["max_think"] = N
                This activates the custom GlobalThinkProcessor, forcing the
                model to emit the '</think>' token sequence after N generated
                tokens. Purely a vLLM-side feature.

            return_token_ids:
                If True, injects:
                    extra_body["return_token_ids"] = True
                The vLLM OpenAI-compatible API (>= v0.10.2) will return:
                    - resp.prompt_token_ids
                    - resp.choices[*].token_ids
                allowing drift-free RL training by exposing the *exact* tokens
                the model consumed and produced.

            model, messages, sampling:
                Standard OpenAI-compatible chat completion fields. Sampling
                options are created from SamplingConfig and passed through
                untouched.

        Returns:
            (ChatResponse, info):
                ChatResponse contains:
                    .text
                    .completion_token_ids (may be None)
                    .prompt_token_ids (may be None)
                    .finish_reason
                'info' contains raw transport details and args actually sent.
        """

        # Sampling → OpenAI kwargs
        request_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
        )
        request_kwargs.update(sampling.to_openai_kwargs())

        # ----------------------------------------------------------
        # vLLM-specific extensions live under `extra_body`:
        #
        #   - interrupt_thinking -> extra_body["vllm_xargs"]["max_think"]
        #   - return_token_ids   -> extra_body["return_token_ids"] = True
        #
        # Any SamplingConfig.extras may also inject/override fields by
        # providing an "extra_body" dict.
        # ----------------------------------------------------------
        extra_body: Dict[str, Any] = {}

        # Merge any existing extras (SamplingConfig.extras → extra_body)
        existing_extra_body = request_kwargs.pop("extra_body", None)
        if isinstance(existing_extra_body, dict):
            extra_body.update(existing_extra_body)

        # Extract or create vllm_xargs
        vllm_xargs = extra_body.get("vllm_xargs", {})

        # Think forcing
        if interrupt_thinking is not None:
            if not isinstance(interrupt_thinking, int) or interrupt_thinking <= 0:
                raise ValueError("interrupt_thinking must be a positive integer")
            vllm_xargs["max_think"] = interrupt_thinking

        if vllm_xargs:
            extra_body["vllm_xargs"] = vllm_xargs

        # Token IDs
        if return_token_ids:
            extra_body["return_token_ids"] = True

        if extra_body:
            request_kwargs["extra_body"] = extra_body

        # ----------------------------------------------------------
        # Perform inference
        # ----------------------------------------------------------
        resp = await self._async_client.chat.completions.create(**request_kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason

        # Extract token IDs if present
        prompt_token_ids = getattr(resp, "prompt_token_ids", None)
        completion_token_ids = getattr(choice, "token_ids", None)

        chat_resp = ChatResponse(
            text=text,
            finish_reason=finish_reason,
            completion_token_ids=completion_token_ids,
            prompt_token_ids=prompt_token_ids,
        )

        info: Dict[str, Any] = {
            "raw_response": resp.model_dump(exclude_none=True),
            "used_args": request_kwargs,
        }

        return chat_resp, info

    # ---- ChatClient.sync_weights --------------------------

    def sync_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        reset_cache: bool = True,
        version: Optional[str] = None,
    ) -> str:
        """
        Push updated model parameters into the running vLLM server.

        Optimized batch implementation:
            1. Sort keys to ensure deterministic ordering.
            2. Send ONE metadata payload (POST /update_param_batch).
            3. Stream tensors via NCCL broadcast loop.
            4. Barrier & Finalize.

        Returns:
            version string (either supplied or autogenerated)
        """

        if self._pynccl_comm is None or self._rank is None:
            if not self.enable_weight_updates:
                raise RuntimeError(
                    "sync_weights() called on inference-only client "
                    "(enable_weight_updates=False)."
                )
            raise RuntimeError("Communicator not initialized.")

        start = time.time()

        # 1. Prepare Metadata
        # We must iterate in a stable order so the server and client
        # broadcast/recv the same tensors in the same order.
        sorted_keys = sorted(params.keys())
        metadata: List[Dict[str, Any]] = []

        for name in sorted_keys:
            tensor = params[name]
            metadata.append(
                {
                    "name": name,
                    "dtype": str(tensor.dtype),
                    "shape": tuple(tensor.shape),
                }
            )

        # 2. Control Plane: Announce Batch
        url = f"{self.server_url}/update_param_batch"
        try:
            resp = self._session.post(
                url,
                json={"metadata": metadata},
                timeout=timeout_s,
            )
        except Timeout:
            raise TimeoutError("HTTP timeout during batch metadata send")
        except Exception as exc:
            raise RuntimeError(f"Error sending batch metadata: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Server rejected update_param_batch: {resp.status_code} {resp.text}"
            )

        # 3. Data Plane: Stream Tensors
        for name in sorted_keys:
            tensor = params[name]
            # Ensure contiguous memory before broadcast
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            self._pynccl_comm.broadcast(tensor, src=self._rank)

        # 4. Synchronization
        self._pynccl_comm.group.barrier()

        if (time.time() - start) > timeout_s:
            raise TimeoutError(f"sync_weights exceeded {timeout_s}s")

        # Note: The server-side /update_param_batch endpoint handles the
        # cache reset automatically at the end of the RPC call.
        if reset_cache:
            # We can still call this explicitly if desired, or rely on server.
            # Calling it again is harmless but adds a small RTT.
            # For strict backward compat, we leave it, or we rely on the server side.
            pass

        # Wait for server background weight-update tasks to drain
        while self.get_num_background_tasks() > 0:
            time.sleep(0.2)

        return version or f"vllm-{int(time.time())}"

    # ---- Control-plane helpers ---------------------------------

    def _check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Poll /health until the server responds OK or timeout expires.
        Ensures we don't start NCCL or inference before the server is alive.
        """
        url = f"{self.server_url}/health"
        start_time = time.time()

        while True:
            try:
                r = self._session.get(url, timeout=5.0)
                if r.status_code == 200:
                    log.info("vLLM server is up")
                    return
            except RequestException:
                pass

            if total_timeout and (time.time() - start_time) >= total_timeout:
                raise ConnectionError(
                    f"vLLM server not reachable at {self.host}:{self.port} "
                    f"after {total_timeout} seconds"
                )

            log.info("vLLM server not ready, retrying...")
            time.sleep(retry_interval)

    def _init_communicator(self) -> None:
        """
        Establish the client's NCCL communicator:
          * query world size from server
          * tell server workers to initialize their communicator
          * create client-side NCCL process group
        """

        # 1) query world size
        r = self._session.get(f"{self.server_url}/get_world_size", timeout=10.0)
        r.raise_for_status()
        vllm_world_size = r.json()["world_size"]
        world_size = vllm_world_size + 1  # client is the extra rank
        self._rank = vllm_world_size

        # 2) ask server workers to init their communicators
        r = self._session.post(
            f"{self.server_url}/init_communicator",
            json={"host": self.host, "port": self.group_port, "world_size": world_size},
            timeout=30.0,
        )
        r.raise_for_status()

        time.sleep(0.1)  # let server initialize NCCL

        # 3) create the matching client-side communicator
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.group_port,
            rank=self._rank,
            world_size=world_size,
        )
        self._pynccl_comm = PyNcclCommunicator(pg, device=self.device)

    def reset_prefix_cache(self) -> None:
        r = self._session.post(f"{self.server_url}/reset_prefix_cache", timeout=30.0)
        r.raise_for_status()

    def get_num_background_tasks(self) -> int:
        r = self._session.post(
            f"{self.server_url}/get_num_background_tasks", timeout=10.0
        )
        r.raise_for_status()
        return r.json()["num_background_tasks"]

    def close_communicator(self) -> None:
        try:
            r = self._session.post(
                f"{self.server_url}/close_communicator", timeout=10.0
            )
            if r.status_code != 200:
                log.warning(
                    "close_communicator responded with %s %s",
                    r.status_code,
                    r.text,
                )
        except ConnectionError:
            # server may already be down — nothing to do.
            pass
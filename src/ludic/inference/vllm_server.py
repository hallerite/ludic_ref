import asyncio
import os
import signal
from argparse import Namespace
from typing import Any, Awaitable, Sequence, Set, Optional, Tuple

# Use V1 engine explicitly.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"

import torch
import uvloop
from fastapi import FastAPI, Request
from vllm import SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs

# V1 logits-processor interface
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor as V1LogitsProcessor,
    BatchUpdate,
    MoveDirectionality,
)

# ---------------------------------------------------------------------------
# Global state for weight updates & background tasks
# ---------------------------------------------------------------------------

MAX_CONCURRENT_WEIGHT_UPDATES = 10
weight_update_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEIGHT_UPDATES)

background_tasks: Set[asyncio.Task[Any]] = set()

RUNTIME_VERSION: int = 0
RUNTIME_VERSION_LOCK = asyncio.Lock()


def create_background_task(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    """Create an async task and track it so we can wait/cancel on shutdown."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


# ---------------------------------------------------------------------------
# Worker extension: NCCL-based weight sync
# ---------------------------------------------------------------------------


class WeightSyncWorkerExtension:
    """
    vLLM worker extension for weight synchronization.

    Each worker:
      - joins a StatelessProcessGroup (TCP)
      - wraps it in a PyNcclCommunicator (NCCL)
      - receives updated weights via broadcast() from the client rank
      - calls `model_runner.model.load_weights` with the new tensors
    """

    pynccl_comm: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: torch.device | None = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Called via engine.collective_rpc on all workers.
        Creates the NCCL communicator for weight updates.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. "
                "Call close_communicator() first."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        assert self.device is not None, "WeightSyncWorkerExtension.device must be set"
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        # client rank is the last rank in the world (host process)
        self.client_rank = world_size - 1

        # --- DEBUG: Print internal vLLM parameter names ---
        # This executes on the worker process. We use Rank 0 to avoid duplicates.
        if self.pynccl_comm.rank == 0:
            print("\n" + "="*60)
            print("🔍 [DEBUG] vLLM Internal Parameter Names (Worker Rank 0)")
            print("="*60)
            try:
                # Access the underlying torch model
                model_instance = self.model_runner.model
                count = 0
                for name, _ in model_instance.named_parameters():
                    print(f"   • {name}")
                    count += 1
                print(f"Total parameters found: {count}")
            except Exception as e:
                print(f"⚠️ Could not print parameter names: {e}")
            print("="*60 + "\n")
        # --------------------------------------------------

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Called via engine.collective_rpc on all workers.
        Receives a single parameter tensor via NCCL broadcast and loads it.
        """
        if self.pynccl_comm is None or self.client_rank is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        torch_dtype = getattr(torch, dtype.split(".")[-1])
        assert self.device is not None
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        # vLLM model runner will apply the incoming weights
        self.model_runner.model.load_weights(weights=[(name, weight)])  # type: ignore[attr-defined]

    def update_param_batch(
        self, metadata_list: Sequence[Tuple[str, str, Sequence[int]]]
    ) -> None:
        """
        Called via engine.collective_rpc on all workers.
        Iterates through the list, creating empty tensors and receiving broadcasts.
        """
        if self.pynccl_comm is None or self.client_rank is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        assert self.device is not None

        for name, dtype_str, shape_list in metadata_list:
            torch_dtype = getattr(torch, dtype_str.split(".")[-1])
            shape = tuple(shape_list)

            # allocate empty on GPU
            weight = torch.empty(shape, dtype=torch_dtype, device=self.device)

            # NCCL Receive
            self.pynccl_comm.broadcast(weight, src=self.client_rank)

            # Apply
            # vLLM model runner will apply the incoming weights
            self.model_runner.model.load_weights(weights=[(name, weight)])  # type: ignore[attr-defined]

        # Barrier to ensure all workers are done
        self.pynccl_comm.group.barrier()

    def close_communicator(self) -> None:
        """
        Called via engine.collective_rpc to tear down communicator state.
        """
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


# ---------------------------------------------------------------------------
# Custom logits processor: inject "</think>" after N tokens (pure V1)
# ---------------------------------------------------------------------------


class GlobalThinkProcessor(V1LogitsProcessor):
    """
    Single V1 logits processor instance per worker.

    For each request in the batch:
      - On BatchUpdate.added, we inspect SamplingParams.extra_args["max_think"].
      - If present and > 0, we remember:
          * a live reference to that request's output_ids (list[int])
          * its trigger_len
      - On apply(logits), we walk each row and, if that request is within
        its think-window, we force the next token of the '</think>' sequence.

    In other words: we force the model to emit '</think>' after a
    user-chosen number of generated tokens.
    """

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # Per-request state: req_idx -> {"output_ids": list[int], "trigger_len": int}
        self.req_state: dict[int, dict[str, Any]] = {}
        # Pre-tokenized think_ids injected into vllm_config BEFORE engine spawn
        self.think_ids = vllm_config.additional_config.get("think_ids", [])

    # ---- required by V1 interface ----

    def is_argmax_invariant(self) -> bool:
        # We overwrite logits and hence argmax, so no.
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Called whenever the persistent batch changes (add/remove/move),
        *before* each forward pass.
        """
        if batch_update is None:
            return

        # 1) Handle removals
        for ridx in batch_update.removed:
            if ridx in self.req_state:
                self.req_state.pop(ridx, None)

        # 2) Handle additions
        for (req_idx, params, prompt_ids, output_ids) in batch_update.added:
            assert isinstance(params, SamplingParams)
            extra_args = getattr(params, "extra_args", None)

            trigger_len = None
            if isinstance(extra_args, dict):
                trigger_len = extra_args.get("max_think")

            if not isinstance(trigger_len, int) or trigger_len <= 0:
                self.req_state.pop(req_idx, None)
                continue

            self.req_state[req_idx] = {
                "output_ids": output_ids,
                "trigger_len": trigger_len,
            }

        # 3) Handle moves
        for (src, dst, direction) in batch_update.moved:
            if direction == MoveDirectionality.UNIDIRECTIONAL:
                state = self.req_state.pop(src, None)
                if state is not None:
                    self.req_state[dst] = state
            else:
                s1 = self.req_state.get(src)
                s2 = self.req_state.get(dst)
                if s1 is not None or s2 is not None:
                    self.req_state[src], self.req_state[dst] = s2, s1

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch_size, vocab_size]
        We mutate rows in-place where we want to force '</think>'.
        """
        if not self.think_ids or not self.req_state:
            return logits

        batch_size = logits.shape[0]
        think_ids = self.think_ids

        for req_idx in range(batch_size):
            state = self.req_state.get(req_idx)
            if state is None:
                continue

            output_ids: list[int] = state["output_ids"]
            trigger_len: int = state["trigger_len"]

            seq_len = len(output_ids)
            pos = seq_len - trigger_len

            if pos < 0 or pos >= len(think_ids):
                continue

            forced_id = think_ids[pos]

            row = logits[req_idx]
            row.fill_(float("-inf"))
            row[forced_id] = 0.0

        return logits


# ---------------------------------------------------------------------------
# Server / app setup
# ---------------------------------------------------------------------------


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    # ----------------------------------------------------------------------
    # 1) Build engine_args from CLI and inject our extension + logits proc
    # ----------------------------------------------------------------------
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Wire worker extension
    worker_ext = "ludic.inference.vllm_server.WeightSyncWorkerExtension"
    engine_args.worker_extension_cls = worker_ext

    # Wire our GlobalThinkProcessor into the engine-wide logits processor list.
    # If user already passed --logits-processors, append ours.
    think_proc = "ludic.inference.vllm_server:GlobalThinkProcessor"
    if engine_args.logits_processors:
        if think_proc not in engine_args.logits_processors:
            engine_args.logits_processors.append(think_proc)
    else:
        engine_args.logits_processors = [think_proc]

    # ----------------------------------------------------------------------
    # 2) Build VllmConfig from engine_args (now containing logits_processors)
    # ----------------------------------------------------------------------
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # --------------------------------------------------------------
    # 3) Pre-tokenize '</think>' using the *same* tokenizer config
    #    the engine will use. This is controller-side only.
    # --------------------------------------------------------------
    try:
        tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config
        )
        think_ids = tokenizer.encode("</think>", add_special_tokens=False)
        vllm_config.additional_config["think_ids"] = think_ids
    except Exception as e:
        raise RuntimeError(
            f"Failed to pre-tokenize '</think>' for model "
            f"{vllm_config.model_config.model}: {e}"
        ) from e

    # ----------------------------------------------------------------------
    # 4) Build AsyncLLM engine from the prepared config.
    #    At this point, vllm_config already knows about GlobalThinkProcessor
    #    and think_ids. V1 will instantiate our logits processor on workers.
    # ----------------------------------------------------------------------
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    app: FastAPI = build_app(args)

    # ------------------------ control-plane endpoints ---------------------

    # TODO: override
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def get_world_size() -> dict[str, int]:
        return {
            "world_size": args.tensor_parallel_size * args.data_parallel_size
        }

    @app.get("/runtime_version")
    async def runtime_version() -> dict[str, int]:
        return {"version": RUNTIME_VERSION}

    @app.post("/init_communicator")
    async def init_communicator(request: Request) -> dict[str, str]:
        """
        Client tells all workers to join a weight-sync process group.
        """
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")

        create_background_task(
            engine.collective_rpc(
                "init_communicator", args=(host, port, world_size)
            )
        )
        return {"status": "ok"}

    @app.post("/update_named_param")
    async def update_named_param(request: Request) -> dict[str, str]:
        """
        Update a single named parameter.

        Client side:
          1) POST name/dtype/shape here
          2) Immediately run NCCL broadcast(weights, src=client_rank)

        Worker side:
          - allocates empty tensor of given shape/dtype
          - calls broadcast(empty, src=client_rank)
          - loads the received tensor into the model
        """
        data = await request.json()
        name = data.get("name")
        dtype = data.get("dtype")
        shape = data.get("shape")
        shape_tuple = tuple(shape)

        async def throttled_update() -> None:
            async with weight_update_semaphore:
                await engine.collective_rpc(
                    "update_named_param", args=(name, dtype, shape_tuple)
                )

        create_background_task(throttled_update())
        return {"status": "ok"}

    @app.post("/update_param_batch")
    async def update_param_batch(request: Request) -> dict[str, str]:
        """
        Receives the batch metadata manifest.
        Triggers the worker extension to enter the receiving loop.
        """
        data = await request.json()
        metadata = data.get("metadata", [])  # List of {name, dtype, shape}
        
        # Check if an explicit version was provided by the Trainer
        forced_version = data.get("version")

        # Convert dicts to tuples for RPC serialization safety
        # (name, dtype, shape)
        rpc_args = [(m["name"], m["dtype"], m["shape"]) for m in metadata]

        async def do_update_batch() -> None:
            async with weight_update_semaphore:
                # This RPC call will block the workers in the receiving loop
                # until the client finishes broadcasting all tensors.
                await engine.collective_rpc("update_param_batch", args=(rpc_args,))

                # Reset cache and bump version after full batch
                await engine.reset_prefix_cache()
                
                global RUNTIME_VERSION
                async with RUNTIME_VERSION_LOCK:
                    if forced_version is not None:
                        RUNTIME_VERSION = int(forced_version)
                    else:
                        RUNTIME_VERSION += 1

        create_background_task(do_update_batch())
        return {"status": "ok"}

    @app.post("/sync_weights")
    async def sync_weights(request: Request) -> dict[str, Any]:
        """
        Optional batched update endpoint.

        Body: { "params": [ {name, dtype, shape}, ... ], "version": "optional-tag" }

        Semantics:
          - Schedules a background task that calls update_named_param for each param.
          - Bumps RUNTIME_VERSION once all workers have processed the batch.
          - HTTP returns immediately after scheduling; client can poll
            /get_num_background_tasks or /runtime_version if it wants to wait.
        """
        data = await request.json()
        params = data.get("params", [])
        requested_version = data.get("version")

        async def do_update() -> None:
            async with weight_update_semaphore:
                for p in params:
                    name = p["name"]
                    dtype = p["dtype"]
                    shape = tuple(p["shape"])
                    await engine.collective_rpc(
                        "update_named_param", args=(name, dtype, shape)
                    )
                
                global RUNTIME_VERSION
                async with RUNTIME_VERSION_LOCK:
                    if requested_version is not None:
                        try:
                            RUNTIME_VERSION = int(requested_version)
                        except ValueError:
                            # If version is a string (e.g. "v1"), just increment
                            RUNTIME_VERSION += 1
                    else:
                        RUNTIME_VERSION += 1

        create_background_task(do_update())
        return {"status": "ok", "requested_version": requested_version}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, str]:
        """
        Reset any KV/prefix caches on the engine.
        """
        create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks() -> dict[str, int]:
        return {"num_background_tasks": len(background_tasks)}

    @app.post("/close_communicator")
    async def close_communicator() -> dict[str, str]:
        """
        Tear down NCCL communicator on all workers.
        """
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}

    # ------------------------ start HTTP server --------------------------

    vllm_config_live = await engine.get_vllm_config()
    print(vllm_config_live)

    await init_app_state(engine, vllm_config_live, app.state, args)

    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    # graceful shutdown of background tasks
    for task in list(background_tasks):
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    sock.close()


def main() -> None:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with weight synchronization"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args() or Namespace()
    validate_parsed_serve_args(args)
    print(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
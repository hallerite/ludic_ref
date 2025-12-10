import time
import torch
from typing import List, Optional
from ludic.distributed.interfaces import ControlPlane, WeightMetadata, TensorCommunicator
from ludic.inference.vllm_client import VLLMChatClient
from ludic.distributed.publisher import BroadcastPolicyPublisher

class VllmControlPlane(ControlPlane):
    def __init__(self, client: VLLMChatClient):
        self.client = client
        self.session = client._session
        self.url = client.server_url

    def announce_update_batch(self, metadata: List[WeightMetadata], version: Optional[int] = None) -> None:
        """
        Hits the /update_param_batch endpoint to prepare the server.
        """
        payload = {"metadata": metadata}
        if version is not None:
            payload["version"] = version

        # The server endpoint returns immediately after scheduling the task.
        # We ensure the server is listening via the HTTP response before broadcasting.
        resp = self.session.post(
            f"{self.url}/update_param_batch",
            json=payload,
            timeout=30.0
        )
        resp.raise_for_status()

    def finalize_update(self, version: str | None = None) -> None:
        """
        Polls the server to ensure background weight application and 
        cache resets are complete before allowing training to proceed.
        """
        while self.client.get_num_background_tasks() > 0:
            time.sleep(0.2)

class VllmTensorCommunicator(TensorCommunicator):
    def __init__(self, client: VLLMChatClient):
        if not client._pynccl_comm:
            raise RuntimeError("vLLM Client has no active NCCL communicator")
        self._comm = client._pynccl_comm
        self._rank = client._rank

    @property
    def rank(self) -> int:
        return self._rank

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self._comm.broadcast(tensor, src=src)

    def barrier(self) -> None:
        self._comm.group.barrier()

def create_vllm_publisher(client: VLLMChatClient) -> BroadcastPolicyPublisher:
    """
    Helper to wire up a PolicyPublisher from an existing VLLM client.
    
    Usage:
        client = VLLMChatClient(enable_weight_updates=True)
        publisher = create_vllm_publisher(client)
        trainer = Trainer(..., publisher=publisher)
    """
    control = VllmControlPlane(client)
    comm = VllmTensorCommunicator(client)
    # The client acts as the source rank (the orchestrator)
    return BroadcastPolicyPublisher(control, comm, src_rank=comm.rank)
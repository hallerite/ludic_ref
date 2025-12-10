import torch
from typing import Mapping, List, Optional
from ludic.distributed.interfaces import (
    PolicyPublisher, 
    ControlPlane, 
    TensorCommunicator, 
    WeightMetadata
)

class BroadcastPolicyPublisher(PolicyPublisher):
    """
    Publishes weights by:
    1. Sending a metadata manifest via the Control Plane (HTTP).
    2. Streaming the tensors via the Data Plane (NCCL).
    """
    def __init__(
        self, 
        control: ControlPlane, 
        comm: TensorCommunicator,
        src_rank: int
    ):
        self.control = control
        self.comm = comm
        self.src_rank = src_rank

    def publish(self, state_dict: Mapping[str, torch.Tensor], version: Optional[int] = None) -> None:
        # 1. Prepare Metadata
        metadata: List[WeightMetadata] = []
        # We need a stable order for the loop
        sorted_keys = sorted(state_dict.keys())
        
        for name in sorted_keys:
            tensor = state_dict[name]
            metadata.append({
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": tuple(tensor.shape)
            })

        # 2. Control Plane: Announce the Batch
        # This tells the server: "Allocate these tensors and get ready to receive"
        # We pass the version here so the server knows what "time" it is.
        self.control.announce_update_batch(metadata, version=version)

        # 3. Data Plane: Stream Tensors
        # The server is now in a loop waiting for broadcast calls matching the metadata list
        for name in sorted_keys:
            tensor = state_dict[name]
            # Ensure tensor is contiguous and on the right device before sending
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            self.comm.broadcast(tensor, src=self.src_rank)

        # 4. Synchronization
        self.comm.barrier()
        
        # 5. Finalize (e.g., reset prefix cache, bump version)
        self.control.finalize_update()
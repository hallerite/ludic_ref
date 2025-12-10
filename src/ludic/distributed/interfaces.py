from typing import Protocol, Mapping, List, TypedDict, runtime_checkable, Optional
import torch

@runtime_checkable
class TensorCommunicator(Protocol):
    """
    The Data Plane: Responsible for moving raw tensor bytes.
    Implementations: PyNcclCommunicator, Gloo, etc.
    """
    @property
    def rank(self) -> int: 
        """The rank of this process in the distributed group."""
        ...

    def broadcast(self, tensor: torch.Tensor, src: int) -> None: 
        """Broadcast a tensor from src rank to all others."""
        ...

    def barrier(self) -> None: 
        """Block until all processes reach this point."""
        ...

class WeightMetadata(TypedDict):
    """Schema for the metadata sent via the Control Plane."""
    name: str
    dtype: str
    shape: tuple[int, ...]

@runtime_checkable
class ControlPlane(Protocol):
    """
    The Control Plane: Signals the runtime to prepare for updates.
    Implementations: VllmControlPlane (HTTP), RayControlPlane (RPC).
    """
    def announce_update_batch(self, metadata: List[WeightMetadata], version: Optional[int] = None) -> None:
        """
        Send ONE batch request containing metadata for all params to be updated.
        The runtime should prepare empty tensors and wait for NCCL broadcast.
        
        Args:
            metadata: List of tensor specs.
            version: Optional explicit version number to set on the runtime.
        """
        ...
    
    def finalize_update(self, version: str | None = None) -> None:
        """Signal that the broadcast is complete and caches can be reset."""
        ...

@runtime_checkable
class PolicyPublisher(Protocol):
    """
    The Orchestrator: The only thing the Trainer talks to.
    """
    def publish(self, state_dict: Mapping[str, torch.Tensor], version: Optional[int] = None) -> None:
        """
        Publishes the state_dict to the remote inference engine.
        
        Args:
            state_dict: The model parameters to push.
            version: Optional explicit version number (clock) for PipelineRL.
        """
        ...
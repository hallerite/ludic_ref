from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    """
    High-level trainer configuration.

    This is *purely* about optimization / model device / collation.
    Rollout and batch-generation config live in BatchSource / Orchestrator.

    ==========================
    Model / Optimization
    ==========================

    - model_device:
          Device where the trainable model lives ("cuda:0", "cpu", ...).
          Trainer will move *batches* to this device, but assumes the model
          is already on the correct device and/or FSDP-wrapped.

    - runtime_device:
          Device to move tensors to before pushing them into the serving
          runtime. If None, defaults to `model_device`.

    - lr, weight_decay, betas, eps:
          AdamW hyperparameters.

    - max_grad_norm:
          Gradient clipping threshold; None disables clipping.
          
    - grad_accum_steps:
          Number of micro-batches to accumulate gradients over before 
          performing one optimizer step (the 'macro-step' size).
          
    - sync_every_steps:
          Frequency (in macro-steps) at which to push updated policy 
          weights to the Agent's runtime (e.g., vLLM).

    - mixed_precision_dtype:
          Optional string to configure FSDP's mixed precision policy. 
          Use "bf16" or "fp16". If None, defaults to full precision (fp32).

    ==========================
    Collation
    ==========================

    - pad_token_id:
          Used when padding sequences during SAW collation.

    ==========================
    Distributed
    ==========================

    - reduce_stats_across_ranks:
          If True (and torch.distributed is initialized), Trainer will all-reduce
          the per-rank stats dict before logging/returning it.
    """

    # ----- model / optimization -------------------
    model_device: str = "cuda"
    runtime_device: Optional[str] = None

    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    max_grad_norm: Optional[float] = 1.0

    # FSDP/RLHF specific settings
    grad_accum_steps: int = 16
    sync_every_steps: int = 1
    mixed_precision_dtype: Optional[str] = "bf16"

    # PipelineRL specific settings
    max_lag: Optional[int] = None  # Drop batches older than N steps
    reduce_stats_across_ranks: bool = False

    # ----- collation ------------------------------
    pad_token_id: int = 0

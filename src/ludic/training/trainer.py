from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Mapping

import torch
from torch import nn, optim, Tensor
import torch.distributed as dist
from torch.distributed import fsdp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from ludic.distributed.interfaces import PolicyPublisher
from ludic.training.checkpoint import CheckpointConfig, CheckpointManager
from ludic.training.algorithm import RLAlgorithm
from ludic.training.loggers import TrainingLogger
from ludic.training.config import TrainerConfig
from ludic.training.stats import aggregate_stats, Reducer
from ludic.training.types import SAWBatch, SAWItem, BatchSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collation: SAWItems -> tensor batch
# ---------------------------------------------------------------------------

def _collate_saw_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Collate a list of SAWItem into a simple dense batch of tensors.

    - Left-aligns sequences and pads to max length in this batch.
    - Returns a dict suitable for RLAlgorithm.loss.compute():

          {
              "input_ids":      [B, T] long,
              "attention_mask": [B, T] long,
              "action_mask":    [B, T] float,
              "weight":         [B]    float,
          }

    If you want microbatching or a DataLoader, you can refactor this later.
    """
    if not items:
        raise ValueError("Cannot collate empty list of SAWItems")

    lengths = [len(it.input_ids) for it in items]
    max_len = max(lengths)

    input_ids_list: List[Tensor] = []
    attn_mask_list: List[Tensor] = []
    action_mask_list: List[Tensor] = []
    weights_list: List[Tensor] = []

    for it in items:
        L = len(it.input_ids)

        ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
        am = torch.zeros((max_len,), dtype=torch.long)
        actm = torch.zeros((max_len,), dtype=torch.float32)

        ids[:L] = torch.tensor(it.input_ids, dtype=torch.long)
        am[:L] = torch.tensor(it.attention_mask, dtype=torch.long)
        actm[:L] = torch.tensor(it.action_mask, dtype=torch.float32)

        input_ids_list.append(ids)
        attn_mask_list.append(am)
        action_mask_list.append(actm)
        weights_list.append(torch.tensor(it.weight, dtype=torch.float32))

    batch = {
        "input_ids": torch.stack(input_ids_list, dim=0).to(device),
        "attention_mask": torch.stack(attn_mask_list, dim=0).to(device),
        "action_mask": torch.stack(action_mask_list, dim=0).to(device),
        "weight": torch.stack(weights_list, dim=0).to(device),
    }
    return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates the training loop with gradient accumulation:

        Loop `grad_accum_steps`:
            BatchSource.next_batch() → micro-batch
              ↓
            collate → tensors
              ↓
            RLAlgorithm.compute_loss(model, batch)
              ↓
            scaled_loss.backward() (with gradient sync disabled on non-final micros for FSDP)
          ↓
        optimizer.step()
          ↓
        optimizer.zero_grad()  # <-- Grads freed
          ↓
        publisher.publish(...)     # <-- Sync

    Trainer is agnostic to envs, contexts, rollouts, and tokenization.

    This variant is FSDP2-aware:

      - If `model` is wrapped in FSDP2 (fully_shard), it will:
          * disable gradient sync on non-final micro-batches
          * gather a full state dict on rank 0 via DCP APIs
          * push the full (unsharded) params to the runtime

      - On non-FSDP models, it just uses `named_parameters()` as before.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        publisher: PolicyPublisher,
        cfg: TrainerConfig = TrainerConfig(),
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
        enable_gradient_checkpointing: bool = False,
        checkpointer: Optional[CheckpointManager] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        resume_from: Optional[int | str] = None,
        train_logger: Optional[TrainingLogger] = None,
        reducers: Optional[Mapping[str, Reducer]] = None,
    ) -> None:
        """
        Args:
            model:
                Trainable policy model (typically a HF CausalLM-like module).
                Must accept (input_ids, attention_mask) and expose .logits.
                May be wrapped in FSDP.

            algo:
                RLAlgorithm = (CreditAssigner + Loss).

            batch_source:
                Any object implementing BatchSource.next_batch() -> SAWBatch.
                This is where rollouts, replay, branching, curricula live.

            publisher:
                Abstract interface to push weights to inference workers.

            cfg:
                TrainerConfig for device, optimizer hyperparams, pad_token_id,
                grad_accum_steps, and sync_every_steps.

            param_filter:
                Optional predicate (name, Tensor) -> bool deciding which
                parameters get pushed into the runtime.

            enable_gradient_checkpointing:
                If True, enables activation checkpointing on the model to save VRAM.
                Also automatically disables KV cache and enables input gradients.

            checkpointer:
                Optional CheckpointManager for periodic HF-format saves. If not
                provided, one will be constructed from `checkpoint_config`.
                Prefer `checkpoint_config` for standard usage; `checkpointer`
                is an escape hatch to inject a custom implementation.

            checkpoint_config:
                Convenience config to build a default CheckpointManager.

            resume_from:
                Optional checkpoint to load on startup. Accepts a step number
                or an explicit checkpoint directory path. Requires a
                checkpointer (explicit or via checkpoint_config).

            train_logger:
                Optional stats logger (e.g., PrintLogger, WandbLogger). Called
                once per train_step with (step, stats).

            reducers:
                Optional aggregation reducers passed to aggregate_stats to
                compute custom metrics from SAWItem meta.
        """
        self.cfg = cfg
        self.train_logger = train_logger
        self.reducers = reducers

        # Assume caller has already done any FSDP wrapping / device placement.
        # We do NOT unconditionally .to(device) for FSDP; that’s the caller’s job.
        self.model = model.to(cfg.model_device) if not isinstance(model, fsdp.FSDPModule) else model  # type: ignore[arg-type]

        self.algo = algo
        self.publisher = publisher
        self._batch_source = batch_source
        self.sync_every_steps = self.cfg.sync_every_steps
        self.param_filter = param_filter
        self._train_step_idx = 0
        self._checkpointer = checkpointer or (
            CheckpointManager(checkpoint_config) if checkpoint_config is not None else None
        )

        # ---- Gradient Checkpointing Setup ----------------------------
        if enable_gradient_checkpointing:
            logger.info("🛡️ Enabling Gradient Checkpointing (Activation Checkpointing)...")
            
            # 1. Enable on the model (HuggingFace standard API)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("⚠️ Model does not have 'gradient_checkpointing_enable' method.")

            # 2. Disable KV Cache (incompatible with checkpointing + training)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

            # 3. Enable input gradients
            # This is often required so that gradients can flow through the checkpointed segments.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # Fallback for generic torch modules if needed, though usually HF specific
                pass
        # --------------------------------------------------------------

        # Initialize optimizer
        self.optimizer = self.initialize_optimizer()

        # Assume gradients are zeroed at init
        self.optimizer.zero_grad(set_to_none=True)

        # Optionally resume from a checkpoint (weights + optimizer)
        if resume_from is not None:
            if self._checkpointer is None:
                raise ValueError("resume_from requires a CheckpointManager or checkpoint_config")
            load_kwargs: Dict[str, Optional[object]] = {}
            if isinstance(resume_from, int):
                load_kwargs["step"] = resume_from
            elif isinstance(resume_from, str):
                # Prefer treating a real path as explicit target; fall back to int-like strings
                if Path(resume_from).exists():
                    load_kwargs["path"] = resume_from
                elif resume_from.isdigit():
                    load_kwargs["step"] = int(resume_from)
            meta = self._checkpointer.load(
                self.model,
                optimizer=self.optimizer,
                step=load_kwargs.get("step"),  # type: ignore[arg-type]
                path=load_kwargs.get("path"),  # type: ignore[arg-type]
            )
            # Default to the saved step, else keep current.
            self._train_step_idx = int(meta.get("step", self._train_step_idx))
            logger.info("✅ Resumed from checkpoint at step %s", self._train_step_idx)

    # ------------------------------------------------------------------
    # Optimizer initialization
    # ------------------------------------------------------------------

    def initialize_optimizer(self) -> optim.Optimizer:
        """
        Initialize and return the optimizer.

        - We create AdamW over all model parameters with hyperparams
          from TrainerConfig:

                lr         = cfg.lr
                weight_decay= cfg.weight_decay
                betas       = cfg.betas
                eps         = cfg.eps
        """
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    def _maybe_reduce_stats_across_ranks(
        self,
        stats: Dict[str, float],
        *,
        device: torch.device,
    ) -> Dict[str, float]:
        if not self.cfg.reduce_stats_across_ranks:
            return stats
        if not (dist.is_available() and dist.is_initialized()):
            return stats

        world_size = dist.get_world_size()
        sum_keys = {
            "num_samples",
            "num_rollouts",
            "total_completion_tokens",
        }
        reduced: Dict[str, float] = {}

        for k, v in stats.items():
            if not isinstance(v, (int, float)):
                continue
            t = torch.tensor(float(v), device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if k in sum_keys:
                reduced[k] = float(t.item())
            else:
                reduced[k] = float(t.item()) / float(world_size)

        # Preserve the local step counter
        reduced["train_step"] = float(self._train_step_idx)
        return reduced

    # ------------------------------------------------------------------
    # Memory utilities (GPU-only)
    # ------------------------------------------------------------------

    def _reset_peak_memory(self, device: torch.device) -> Optional[int]:
        """
        Reset CUDA peak stats and return current allocated bytes.
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)

    def _capture_forward_memory_stats(
        self,
        device: torch.device,
        pre_forward_alloc: Optional[int],
    ) -> tuple[Dict[str, float], Optional[int], Optional[int]]:
        """
        Capture forward-pass peak memory and activation peak.

        Returns: (stats_dict, alloc_after_forward_bytes, forward_peak_bytes)
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return {}, None, None

        torch.cuda.synchronize(device)
        mb = 1024 ** 2

        forward_peak = torch.cuda.max_memory_allocated(device)
        forward_activation_peak = max(0, forward_peak - (pre_forward_alloc or 0))
        alloc_after_forward = torch.cuda.memory_allocated(device)

        stats = {
            "gpu_forward_peak_mb": forward_peak / mb,
            "gpu_forward_activation_peak_mb": forward_activation_peak / mb,
        }

        logger.info(
            "GPU forward memory (MB) peak/activation_peak: %.1f / %.1f",
            stats["gpu_forward_peak_mb"],
            stats["gpu_forward_activation_peak_mb"],
        )

        # Reset for backward measurement
        torch.cuda.reset_peak_memory_stats(device)

        return stats, alloc_after_forward, forward_peak

    def _collect_memory_stats(
        self,
        device: torch.device,
        *,
        baseline_alloc: Optional[int],
    ) -> tuple[Dict[str, float], Optional[int]]:
        """
        Capture current/peak GPU memory and backward activation footprint.

        Returns: (stats_dict, peak_bytes)
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return {}, None

        torch.cuda.synchronize(device)
        mb = 1024 ** 2

        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        backward_activation_peak = max(0, peak - (baseline_alloc or 0))

        stats = {
            "gpu_mem_alloc_mb": alloc / mb,
            "gpu_mem_reserved_mb": reserved / mb,
            "gpu_backward_peak_mb": peak / mb,
            "gpu_backward_activation_peak_mb": backward_activation_peak / mb,
        }

        logger.info(
            "GPU backward memory (MB) alloc/reserved/peak/activation_peak: %.1f / %.1f / %.1f / %.1f",
            stats["gpu_mem_alloc_mb"],
            stats["gpu_mem_reserved_mb"],
            stats["gpu_backward_peak_mb"],
            stats["gpu_backward_activation_peak_mb"],
        )

        return stats, peak

    # ------------------------------------------------------------------
    # Core async train step (now a "macro-step")
    # ------------------------------------------------------------------

    async def train_step(self) -> Dict[str, float]:
        """
        One full "macro" RL step with gradient accumulation:

            - Accumulate gradients over `cfg.grad_accum_steps` micro-batches
            - Perform one optimizer step
            - Free gradients
            - Optionally push updated params into runtime

        Returns:
            Aggregated stats dict from all micro-batches.
        """
        device = torch.device(self.cfg.model_device)
        grad_accum_steps = getattr(self.cfg, "grad_accum_steps", 1)
        if grad_accum_steps < 1:
            grad_accum_steps = 1

        all_micro_stats: List[Dict[str, float]] = []
        all_saw_batches: List[SAWBatch] = []

        # ---- 1) Accumulation Loop (Micro-Steps) ------------------------

        # Note: Gradients are *not* zeroed here. They are accumulated
        # from the previous state (which should be zero).
        self.model.train()

        for micro_step_idx in range(grad_accum_steps):
            
            # ---- 1a) Sample Valid Micro-batch (with Rejection Loop) ----
            # For PipelineRL, we might receive stale data from the queue.
            # We loop until we get a batch containing at least one fresh item.
            while True:
                saw_batch: SAWBatch = await self._batch_source.next_batch()
                
                # If configured, filter out items that exceed max_lag.
                if getattr(self.cfg, "max_lag", None) is not None:
                    current_time = self._train_step_idx
                    limit = self.cfg.max_lag
                    
                    fresh_items = []
                    for item in saw_batch.items:
                        # Default to current_time (0 lag) if tag is missing
                        item_ver = item.meta.get("policy_version", current_time)
                        if (current_time - item_ver) <= limit:
                            fresh_items.append(item)
                    
                    # Update the batch with only fresh items
                    saw_batch.items = fresh_items

                # If the batch is empty (e.g. all stale), drop it and fetch another.
                if not saw_batch.items:
                    continue

                # Batch has valid items, proceed to collation
                break

            all_saw_batches.append(saw_batch)

            # Debug: Check batch size before collation to diagnose OOM or filtering impact
            item_count = len(saw_batch.items)
            logger.info(
                f"[Micro-step {micro_step_idx+1}/{grad_accum_steps}] "
                f"Processing {item_count} SAWItems."
            )

            # ---- 1b) Collate into tensors ------------------------------
            batch_tensors = _collate_saw_items(
                saw_batch.items,
                pad_token_id=self.cfg.pad_token_id,
                device=device,
            )

            # Debug: Check tensor shape [Batch, Time]
            input_shape = batch_tensors["input_ids"].shape
            logger.info(
                f"    -> Collated Tensor Shape: {input_shape} "
                f"(Batch={input_shape[0]}, SeqLen={input_shape[1]})"
            )

            # ---- 1c) FSDP2 gradient sync control ----------------------
            # We only sync (all-reduce) gradients on the *last* micro-batch
            is_last_micro = (micro_step_idx == grad_accum_steps - 1)
            grad_sync_disabled = False
            if isinstance(self.model, fsdp.FSDPModule) and not is_last_micro:
                self.model.set_requires_gradient_sync(False)
                grad_sync_disabled = True

            # ---- 1d) Loss + backward (scaled) --------------------------
            pre_forward_alloc = self._reset_peak_memory(device)
            try:
                loss, stats = self.algo.compute_loss(self.model, batch_tensors)

                # Scale loss for accumulation
                scaled_loss = loss / grad_accum_steps
                # Forward memory stats before backward frees activations
                forward_mem_stats, alloc_after_forward, forward_peak = (
                    self._capture_forward_memory_stats(device, pre_forward_alloc)
                )

                scaled_loss.backward()
            finally:
                if grad_sync_disabled and isinstance(self.model, fsdp.FSDPModule):
                    self.model.set_requires_gradient_sync(True)

            # Attach memory stats (if available) for logging/aggregation
            stats = dict(stats)
            backward_mem_stats, backward_peak = self._collect_memory_stats(
                device,
                baseline_alloc=alloc_after_forward,
            )

            # Compute overall peak/activation across forward+backward
            if forward_peak is not None or backward_peak is not None:
                mb = 1024 ** 2
                total_peak = max(
                    forward_peak or 0,
                    backward_peak or 0,
                )
                activation_total_peak = max(0, total_peak - (pre_forward_alloc or 0))

                stats["gpu_mem_peak_mb"] = total_peak / mb
                stats["gpu_activation_peak_mb"] = activation_total_peak / mb

            stats.update(forward_mem_stats)
            stats.update(backward_mem_stats)

            all_micro_stats.append(stats)

        # ---- 2) Gradient Clipping (after loop) -------------------------
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_grad_norm,
            )

        # ---- 3) Optimizer Step (one step for the macro-batch) ----------
        self.optimizer.step()

        # ---- 4) Free Gradients (as requested) --------------------------
        # Grads are freed *after* step and *before* weight sync
        self.optimizer.zero_grad(set_to_none=True)

        self._train_step_idx += 1

        # ---- 5) Push policy update into runtime ------------------------
        if self._train_step_idx % self.sync_every_steps == 0:
            self._push_weights_to_runtime()

        # ---- 6) Enrich stats -------------------------------------------
        final_stats = aggregate_stats(all_micro_stats, all_saw_batches, reducers=self.reducers)
        final_stats["train_step"] = float(self._train_step_idx)

        final_stats = self._maybe_reduce_stats_across_ranks(final_stats, device=device)

        # ---- 7) Optional Checkpoint ------------------------------------
        if self._checkpointer is not None:
            self._checkpointer.maybe_save(
                self.model,
                optimizer=self.optimizer,
                step=self._train_step_idx,
                metadata={"algorithm": self.algo.name},
            )

        # ---- 8) Optional logging ---------------------------------------
        if self.train_logger is not None:
            try:
                self.train_logger.log(self._train_step_idx, final_stats)
            except Exception:
                logger.exception("Stats logger failed at step %s", self._train_step_idx)

        return final_stats

    # ------------------------------------------------------------------
    # Sync wrappers for non-async callers
    # ------------------------------------------------------------------

    def train_step_sync(self) -> Dict[str, float]:
        """
        Synchronous wrapper around `train_step()`.

        Use this from plain scripts/CLIs. Do NOT call inside an existing
        running event loop.
        """
        return asyncio.run(self.train_step())

    async def train(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Run `num_steps` training iterations.

        Args:
            num_steps:
                Number of trainer steps (each step pulls a fresh SAWBatch).

            log_every:
                Frequency (in steps) at which to call `log_fn`.

            log_fn:
                Optional callback(stats_dict) for logging / progress reporting.
        """
        for _ in range(num_steps):
            stats = await self.train_step()
            if log_fn is not None and (self._train_step_idx % log_every == 0):
                log_fn(stats)

    def train_sync(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Synchronous wrapper around `train(...)`.
        """
        asyncio.run(self.train(num_steps, log_every=log_every, log_fn=log_fn))

    # ------------------------------------------------------------------
    # Weight sync into runtime via Agent (FSDP-aware + LoRA-aware)
    # ------------------------------------------------------------------

    def _push_weights_to_runtime(self) -> None:
        """
        Gather weights (handling FSDP if needed) and publish them.

        If the model is a PEFT/LoRA model, we strictly follow the
        Merge -> Publish -> Unmerge pattern so vLLM receives dense weights
        but training continues on adapters.
        """
        # Helper to get the underlying model if wrapped in FSDP2
        inner_model = self.model

        # 1. Check if this is a LoRA/PEFT model
        #    (We use getattr so we don't need to import peft here)
        merge_fn = getattr(inner_model, "merge_adapter", None)
        unmerge_fn = getattr(inner_model, "unmerge_adapter", None)
        is_peft = callable(merge_fn) and callable(unmerge_fn)

        # 2. If LoRA, merge weights before gathering
        if is_peft:
            merge_fn()

        try:
            rank = 0
            is_distributed = dist.is_available() and dist.is_initialized()
            if is_distributed:
                rank = dist.get_rank()

            runtime_device = torch.device(
                self.cfg.runtime_device or self.cfg.model_device
            )

            # We use the current training step as the authoritative version for PipelineRL
            current_version = self._train_step_idx

            raw_params: Dict[str, Tensor] = {}

            # --- Gather Raw Params (FSDP2 or Standard) ---
            if isinstance(self.model, fsdp.FSDPModule):
                # DCP full_state_dict gathering is collective; all ranks must participate.
                # Use cpu_offload=True to avoid materializing a full copy on every rank.
                options = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                )
                full_state = get_model_state_dict(
                    model=self.model,
                    options=options,
                )
                if is_distributed and rank != 0:
                    return
                for k, v in full_state.items():
                    if self.param_filter is not None and not self.param_filter(k, v):
                        continue
                    raw_params[k] = v.detach().to(runtime_device)
            else:
                # Only rank 0 talks to the runtime in distributed mode
                if is_distributed and rank != 0:
                    return

                # Standard model
                for name, p in self.model.named_parameters():
                    # Optimization: In LoRA, only send what we touched (plus what needs fusion)
                    # If not PEFT, only send requires_grad.
                    # If PEFT (merged), we theoretically need to send the whole base layer 
                    # because it changed.
                    
                    # If standard training: send only requires_grad=True
                    # If LoRA (merged): send everything (because base weights need updating in vLLM)
                    if not is_peft and not p.requires_grad:
                        continue
                        
                    if self.param_filter is not None:
                        if not self.param_filter(name, p):
                            continue

                    raw_params[name] = p.detach().to(runtime_device)

            if not raw_params:
                return

            # --- Publish ---
            # NOTE: We simply hand off the raw state dict and rely on the Publisher to handle fusion and renaming.
            self.publisher.publish(raw_params, version=current_version)

        finally:
            # 3. CRITICAL: Unmerge adapters immediately after publishing.
            #    If we don't do this, the optimizer state will become invalid
            #    (gradients calculated on merged weights vs adapter weights).
            if is_peft:
                unmerge_fn()

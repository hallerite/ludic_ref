from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.distributed import fsdp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """
    Configuration for periodic checkpoints.

    - output_dir:       Root directory where checkpoints are stored.
    - every_n_steps:    Save a checkpoint every N trainer steps (0 disables).
    - max_to_keep:      Keep only the most recent N checkpoints (None = keep all).
    - save_optimizer:   Whether to persist optimizer state alongside the model.
    """

    output_dir: str
    every_n_steps: int = 0
    max_to_keep: Optional[int] = 2
    save_optimizer: bool = True

    def __post_init__(self) -> None:
        if self.every_n_steps < 0:
            raise ValueError("every_n_steps must be >= 0")
        if self.max_to_keep is not None and self.max_to_keep < 1:
            raise ValueError("max_to_keep must be None or >= 1")


class CheckpointManager:
    """
    Small helper that saves Trainer checkpoints in HuggingFace format.

    - Uses `save_pretrained(..., state_dict=...)` when available to stay HF-compatible.
    - Handles FSDP full-state gathering on rank 0 only.
    - Prunes older checkpoints if requested.
    """

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg = cfg
        self.base_dir = Path(cfg.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_save(self, step: int) -> bool:
        """Return True if a checkpoint should be taken at this step."""
        return self.cfg.every_n_steps > 0 and step % self.cfg.every_n_steps == 0

    def maybe_save(
        self,
        model: nn.Module,
        *,
        optimizer: Optional[optim.Optimizer],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save a checkpoint if `should_save(step)` is True. Returns the path to
        the checkpoint directory on the primary rank, otherwise None.
        """
        if not self.should_save(step):
            return None
        return self.save(model, optimizer=optimizer, step=step, metadata=metadata)

    def save(
        self,
        model: nn.Module,
        *,
        optimizer: Optional[optim.Optimizer],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Force a checkpoint save. All ranks participate in state_dict collection;
        only the primary rank writes to disk.
        """
        is_primary = self._is_primary_rank()

        ckpt_dir = self.base_dir / f"step_{step:06d}"
        if is_primary:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        state_dict = self._gather_state_dict(model)
        if is_primary:
            self._save_model(model, ckpt_dir, state_dict)

        if optimizer is not None and self.cfg.save_optimizer:
            optim_state = self._gather_optimizer_state_dict(model, optimizer)
            if is_primary:
                torch.save(optim_state, ckpt_dir / "optimizer.pt")

        if is_primary:
            self._write_metadata(ckpt_dir, step, metadata)
            self._write_latest_pointer(step)
            self._prune_old_checkpoints()

        if is_primary:
            logger.info("💾 Saved checkpoint: %s", ckpt_dir)
            return ckpt_dir
        return None

    def load(
        self,
        model: nn.Module,
        *,
        optimizer: Optional[optim.Optimizer] = None,
        step: Optional[int] = None,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint into `model` (and optionally `optimizer`).

        Args:
            model:      Model instance to populate.
            optimizer:  Optimizer to restore if an optimizer checkpoint exists.
            step:       Specific step number to load. If None, uses `path` or latest.
            path:       Explicit checkpoint directory (overrides `step`).

        Returns:
            Metadata dict saved in trainer_state.json (at minimum {"step": int}).
        """
        ckpt_dir = self._resolve_checkpoint_dir(step=step, path=path)
        if ckpt_dir is None:
            raise FileNotFoundError("No checkpoint found to load.")

        metadata = self._load_model(model, ckpt_dir)
        if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
            optim_state: Dict[str, Any] | None
            if self._is_primary_rank():
                optim_state = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu")
            else:
                optim_state = None
            self._load_optimizer_state_dict(model, optimizer, optim_state)
        return metadata

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_primary_rank(self) -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _gather_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Return a full, CPU-offloaded state dict suitable for HF saving.
        """
        if isinstance(model, fsdp.FSDPModule):
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
            )
            return get_model_state_dict(model=model, options=options)

        # Non-FSDP path
        return model.state_dict()

    def _gather_optimizer_state_dict(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> Dict[str, Any]:
        """
        Return an optimizer state dict suitable for saving alongside model weights.
        """
        if isinstance(model, fsdp.FSDPModule):
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
            )
            return get_optimizer_state_dict(
                model=model,
                optimizers=optimizer,
                options=options,
            )
        return optimizer.state_dict()

    def _load_optimizer_state_dict(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        optim_state: Optional[Dict[str, Any]],
    ) -> None:
        """
        Load an optimizer state dict into `optimizer`.

        For FSDP2, rank0 reads the full optimizer state dict and broadcasts it
        to other ranks, which shard it locally.
        """
        if isinstance(model, fsdp.FSDPModule):
            set_optimizer_state_dict(
                model=model,
                optimizers=optimizer,
                optim_state_dict=optim_state or {},
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return

        if optim_state is None:
            return
        optimizer.load_state_dict(optim_state)

    def _save_model(
        self,
        model: nn.Module,
        ckpt_dir: Path,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """
        Save in HuggingFace format when possible, otherwise fall back to a
        torch state_dict + minimal config stub.
        """
        inner_model = getattr(model, "module", model)
        save_pretrained = getattr(inner_model, "save_pretrained", None)

        if callable(save_pretrained):
            save_pretrained(ckpt_dir, state_dict=state_dict)
        else:
            torch.save(state_dict, ckpt_dir / "pytorch_model.bin")
            self._write_stub_config(inner_model, ckpt_dir)

    def _write_stub_config(self, model: nn.Module, ckpt_dir: Path) -> None:
        """
        Write a minimal config.json so the checkpoint directory mirrors HF layout.
        """
        cfg_path = ckpt_dir / "config.json"
        if cfg_path.exists():
            return

        config_payload = {"model_class": model.__class__.__name__}
        if hasattr(model, "config"):
            try:
                config_payload.update(getattr(model, "config").to_dict())  # type: ignore[call-arg]
            except Exception:
                pass

        cfg_path.write_text(json.dumps(config_payload, indent=2))

    def _write_metadata(
        self,
        ckpt_dir: Path,
        step: int,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        payload: Dict[str, Any] = {"step": step}
        if metadata:
            payload.update(metadata)

        meta_path = ckpt_dir / "trainer_state.json"
        meta_path.write_text(json.dumps(payload, indent=2))

    def _write_latest_pointer(self, step: int) -> None:
        latest = {"last_checkpoint": f"step_{step:06d}", "step": step}
        (self.base_dir / "latest.json").write_text(json.dumps(latest, indent=2))

    def _prune_old_checkpoints(self) -> None:
        """
        Remove older checkpoints beyond `max_to_keep`.
        """
        if self.cfg.max_to_keep is None:
            return

        ckpt_dirs = sorted(
            [p for p in self.base_dir.glob("step_*") if p.is_dir()],
            key=self._extract_step,
        )
        excess = len(ckpt_dirs) - self.cfg.max_to_keep
        for old_dir in ckpt_dirs[:excess]:
            shutil.rmtree(old_dir, ignore_errors=True)

    @staticmethod
    def _extract_step(path: Path) -> int:
        try:
            return int(path.name.split("_")[-1])
        except Exception:
            return -1

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_dir(
        self, *, step: Optional[int], path: Optional[str]
    ) -> Optional[Path]:
        if path is not None:
            p = Path(path)
            return p if p.exists() else None

        if step is not None:
            p = self.base_dir / f"step_{step:06d}"
            return p if p.exists() else None

        latest_path = self.base_dir / "latest.json"
        if latest_path.exists():
            try:
                latest = json.loads(latest_path.read_text())
                last = latest.get("last_checkpoint")
                if last:
                    p = self.base_dir / last
                    if p.exists():
                        return p
            except Exception:
                pass

        # Fallback: pick the max existing step directory
        candidates = sorted(
            [p for p in self.base_dir.glob("step_*") if p.is_dir()],
            key=self._extract_step,
        )
        return candidates[-1] if candidates else None

    def _load_model(self, model: nn.Module, ckpt_dir: Path) -> Dict[str, Any]:
        """
        Load model weights from HF-style checkpoint directory.
        """
        state_dict = self._read_state_dict(ckpt_dir)

        if isinstance(model, fsdp.FSDPModule):
            set_model_state_dict(
                model=model,
                model_state_dict=state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
        else:
            model.load_state_dict(state_dict)

        meta_path = ckpt_dir / "trainer_state.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        return {"step": 0}

    def _read_state_dict(self, ckpt_dir: Path) -> Dict[str, torch.Tensor]:
        """
        Read a state_dict saved by `_save_model`.
        """
        candidates = [
            ckpt_dir / "pytorch_model.bin",
            ckpt_dir / "pytorch_model.pt",
            ckpt_dir / "model.safetensors",
        ]
        for path in candidates:
            if path.exists():
                return torch.load(path, map_location="cpu")
        raise FileNotFoundError(f"No model state found in {ckpt_dir}")

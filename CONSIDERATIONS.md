# Considerations (Notes for Later)

## DCP (Distributed Checkpointing) and Resuming With Different GPU Counts

We use PyTorch **DCP** (“Distributed Checkpointing”) APIs for FSDP2:

- Saving gathers a **full (unsharded)** state dict with `full_state_dict=True` (and `cpu_offload=True`).
- Loading uses `set_model_state_dict(..., full_state_dict=True, broadcast_from_rank0=True)` so rank0 provides the full weights and each rank **re-shards locally**. But this behavior is untested and I intend to explicitly test it.

Implication:

- **Resuming with a different `world_size` / GPU count should work**, because we are saving/loading a **full** state dict (not a sharded DTensor checkpoint).

When it would *not* work:

- If we saved a **sharded** checkpoint (DTensor placements tied to a device mesh), then the checkpoint would generally be coupled to the original sharding/world-size (unless additional resharding logic is implemented).

## Checkpoint Format Notes (HF Sharded Saving)

Our checkpoint code collects a full state dict via DCP, but then writes it using HuggingFace `save_pretrained(..., state_dict=...)` when available (see `src/ludic/training/checkpoint.py`).

Notes:

- `save_pretrained()` may write **sharded** weights for large models (e.g. `model-00001-of-000xx.safetensors` + `model.safetensors.index.json`).
- Our loader (`CheckpointManager._read_state_dict`) supports:
  - single-file `pytorch_model.bin` / `pytorch_model.pt`
  - sharded torch checkpoints via `pytorch_model.bin.index.json`
  - single-file `model.safetensors`
  - sharded safetensors via `model.safetensors.index.json` (requires `safetensors`)

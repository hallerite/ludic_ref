# FSDP2 GSM8K Training (3 GPUs + 1 VLLM GPU)

This example shows how to run Ludic with PyTorch FSDP2 for training while serving inference from a separate vLLM instance.

## Layout
- `train_gsm8k_fsdp2.py`: torchrun entrypoint that wraps Qwen2.5-7B-Instruct with `fully_shard`, trains on GSM8K, and pushes weights to vLLM.

## Assumptions
- 4 GPUs total: GPU0 runs vLLM; GPUs 1–3 run training.
- You start vLLM separately on GPU0 serving `Qwen/Qwen2.5-7B-Instruct`.
- Training runs under `torchrun` with NCCL; vLLM weight pushes use a separate NCCL communicator (`pynccl`) owned by the client. They do not interfere.

## Quickstart (template)
You can run everything with the helper script:
```bash
bash examples/fsdp2_training/run_example.sh
```

1. Start vLLM on GPU0 (example using the bundled server):
   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --port 8000 \
     --max-num-seqs 32
   ```

2. Launch training on GPUs 1–3:
   ```bash
   # Pin training to GPUs 1,2,3 so GPU0 stays free for vLLM
   CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. PYTHONUNBUFFERED=1 uv run torchrun --nproc_per_node=3 \
     examples/fsdp2_training/train_gsm8k_fsdp2.py \
       --model Qwen/Qwen2.5-7B-Instruct \
       --vllm-host 127.0.0.1 \
       --vllm-port 8000 \
       --limit 256 \
       --train-steps 50 \
       --concurrency 4 \
       --batch-size 1 \
       --group-size 8 \
       --log-level INFO \
       --logger print
   ```

3. Checkpoints and logs:
   - Checkpoints: `checkpoints_gsm8k_fsdp2/` (rank0 saves).
   - Rollout logs: `fsdp2_gsm8k_rollouts.rank{RANK}.jsonl`.
   - Rank0 prints basic stats; attach `RichLiveLogger` only on rank0.

## Notes
- Mixed precision: uses `fsdp.MixedPrecisionPolicy` with bf16 params / fp32 reductions.
- Activation checkpointing: enabled by default in the script.
- Gradient sync: uses `set_requires_gradient_sync(False/True)` for accumulation; no `no_sync`.
- Weight publishing: only rank0 gathers a full state dict (DCP full_state_dict) and broadcasts to vLLM over the separate pynccl communicator.
- Sample sharding: GSM8K samples are sharded per rank to avoid duplicates; adjust to your data loader if needed.

Tune `--batch-size`, `--group-size`, and `--train-steps` based on hardware. The script is a scaffold; extend it for eval, better logging, and real hyperparameters.

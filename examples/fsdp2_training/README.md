# FSDP2 MATH Training (3 GPUs + 1 vLLM GPU)

This example shows how to run Ludic with PyTorch FSDP2 for training while serving inference from a separate vLLM instance.

## Layout
- `train_math_fsdp2.py`: MATH training with FSDP2 + strict `<think>...</think> \\boxed{...}` parsing.

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
     --gpu_memory_utilization 0.8 \
     --port 8000 \
     --max-num-seqs 32
   ```

2. Launch training on GPUs 1–3:
   ```bash
   # Pin training to GPUs 1,2,3 so GPU0 stays free for vLLM
   CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. PYTHONUNBUFFERED=1 uv run torchrun --nproc_per_node=3 \
     examples/fsdp2_training/train_math_fsdp2.py \
     --model Qwen/Qwen2.5-7B-Instruct \
     --vllm-host 127.0.0.1 \
     --vllm-port 8000 \
     --limit 2048 \
     --train-steps 50 --group-size 8 \
     --concurrency 11 --batch-size 1 --train-temperature 1.0 \
     --eval-before-start --eval-every 10 --eval-limit 100 \
     --eval-concurrency 32 --eval-temperature 0.0 \
     --eval-max-tokens 1024 \
     --log-level INFO \
     --logger print \
     --rank0-only-output
   ```

3. Checkpoints and logs:
   - Checkpoints: `checkpoints_math_fsdp2/` (rank0 saves).
   - Rollout logs: `fsdp2_math_rollouts.rank{RANK}.jsonl`.
   - Rank0 prints basic stats; attach `RichLiveLogger` only on rank0.

## Notes
- Mixed precision: uses `fsdp.MixedPrecisionPolicy` with bf16 params / fp32 reductions.
- Activation checkpointing: enabled by default in the script.
- Gradient sync: uses `set_requires_gradient_sync(False/True)` for accumulation; no `no_sync`.
- Action parsing: the environment does not “re-parse” model outputs; the protocol’s parser must extract the final answer string used for grading.
- Weight publishing: only rank0 gathers a full state dict (DCP full_state_dict) and broadcasts to vLLM over the separate pynccl communicator.
- Sample sharding: MATH samples are sharded per rank to avoid duplicates; adjust to your data loader if needed.

Tune `--batch-size`, `--group-size`, and `--train-steps` based on hardware. The script is a scaffold; extend it for eval, better logging, and real hyperparameters.

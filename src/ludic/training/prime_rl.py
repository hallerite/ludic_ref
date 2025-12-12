import asyncio
import torch
import shutil
from pathlib import Path
from typing import List
from pydantic import Field

# --- Prime-RL Imports ---
from prime_rl.orchestrator.config import OrchestratorConfig as PrimeOrchestratorConfig
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.types import TrainingExample
from prime_rl.utils.client import (
    setup_clients,
    setup_admin_clients,
    check_health,
    update_weights,
    reload_weights,
    init_nccl_broadcast,
)
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_rollout_dir,
    get_step_path,
    get_latest_ckpt_step,
    clean_exit,
)
from prime_rl.utils.logger import setup_logger

# --- Ludic Imports ---
from ludic.training.types import SAWBatch, BatchSource


# ---------------------------------------------------------------------------
# 1. Configuration Bridge
# ---------------------------------------------------------------------------

class LudicConfig(PrimeOrchestratorConfig):
    """
    Extends Prime-RL's OrchestratorConfig to include Ludic-specific settings.
    """
    # Ludic specific registry keys (used by user code to build BatchSource)
    env_kind: str = Field(
        description="Key in the Ludic EnvRegistry", default="my_env"
    )
    protocol_kind: str = Field(
        description="Key in the Ludic ProtocolRegistry", default="single_agent"
    )

    # Ludic specific rollout settings (free-form kwargs for env/protocol)
    protocol_kwargs: dict = Field(default_factory=dict)
    env_kwargs: dict = Field(default_factory=dict)

    # You can still override defaults like batch_size here if desired
    batch_size: int = 128


# ---------------------------------------------------------------------------
# 2. Data Sink (The Converter)
# ---------------------------------------------------------------------------

class PrimeRLDataSink:
    """
    Converts Ludic SAWBatches (in-memory) to Prime-RL MicroBatches (on-disk).
    """

    def __init__(
        self,
        output_dir: Path,
        num_train_workers: int,
        seq_len: int,
        temperature: float,
    ):
        self.rollout_dir = get_rollout_dir(output_dir)
        self.num_train_workers = num_train_workers
        self.seq_len = seq_len
        self.temperature = temperature

    def save_step(self, saw_batch: SAWBatch, step: int):
        """
        Converts Ludic data -> Prime TrainingExamples -> Prime MicroBatches -> Disk.
        """
        train_examples: List[TrainingExample] = []

        for item in saw_batch.items:
            # Split input_ids based on action_mask (0=prompt, 1=completion)
            input_tensor = torch.tensor(item.input_ids)
            mask_tensor = torch.tensor(item.action_mask)

            # Identify indices
            prompt_indices = (mask_tensor == 0).nonzero(as_tuple=True)[0]
            completion_indices = (mask_tensor == 1).nonzero(as_tuple=True)[0]

            if len(completion_indices) == 0:
                # Skip prompts with no completion (shouldn't happen in SAW)
                continue

            prompt_ids = input_tensor[prompt_indices].tolist()
            completion_ids = input_tensor[completion_indices].tolist()

            # Extract logprobs from meta (populated by VLLM client / Ludic)
            # We assume Ludic put them in item.meta['completion_logprobs'].
            completion_logprobs = item.meta.get("completion_logprobs", [])

            # Padding/Truncation safety for logprobs
            if len(completion_logprobs) < len(completion_ids):
                # Pad with 0.0 if missing (prevents crash; watch data quality)
                completion_logprobs += [0.0] * (
                    len(completion_ids) - len(completion_logprobs)
                )

            ex: TrainingExample = {
                "prompt_ids": prompt_ids,
                "prompt_mask": [0] * len(prompt_ids),
                "completion_ids": completion_ids,
                "completion_mask": [1] * len(completion_ids),
                "completion_logprobs": completion_logprobs[: len(completion_ids)],
                # In Ludic, SAWItem.weight is already the scalar advantage
                "advantage": item.weight,
            }
            train_examples.append(ex)

        # Use Prime-RL's native batch packing (First Fit Decreasing)
        batches_per_gpu = prepare_batch(
            train_examples,
            temperature=self.temperature,
            seq_len=self.seq_len,
            num_train_workers=self.num_train_workers,
        )

        # Save to disk where Trainer looks for it
        step_path = self.rollout_dir / f"step_{step}"
        step_path.mkdir(parents=True, exist_ok=True)

        for i, micro_batches in enumerate(batches_per_gpu):
            # Prime-RL trainer expects rank_{i}.pt
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")

            # Atomic write to avoid partially-written files being read
            torch.save(micro_batches, tmp_path)
            tmp_path.rename(batch_path)


# ---------------------------------------------------------------------------
# 3. The Ludic–Prime Orchestrator
# ---------------------------------------------------------------------------

class PrimeOrchestrator:
    """
    Thin bridge:

    - pulls SAWBatches from a user-provided BatchSource
    - handles Prime-RL async weight sync
    - converts & writes MicroBatches for the Prime-RL trainer
    """

    def __init__(self, config: LudicConfig, batch_source: BatchSource):
        self.config = config
        self.logger = setup_logger(config.log.level)

        # 0. Require a BatchSource
        if batch_source is None:
            raise ValueError("PrimeOrchestrator requires a BatchSource instance.")
        self.batch_source = batch_source

        # 1. Setup infrastructure clients (Prime-RL native)
        self.clients = setup_clients(config.client)
        self.admin_clients = setup_admin_clients(config.client)

        # 2. Setup Data Sink (Prime format)
        self.sink = PrimeRLDataSink(
            output_dir=config.output_dir,
            num_train_workers=config.num_train_workers,
            seq_len=config.seq_len,
            temperature=config.sampling.temperature,
        )

        # 3. State
        self.step = 0
        self.ckpt_step = 0

    async def setup(self):
        """Initialize infrastructure connectivity."""
        self.logger.info("Checking Inference Health...")
        await check_health(self.admin_clients)

        # Initialize NCCL broadcast if configured
        if self.config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.admin_clients,
                self.config.weight_broadcast.host,
                self.config.weight_broadcast.port,
                self.config.weight_broadcast.timeout,
            )

        # Reset to base model
        self.logger.info("Resetting to base model...")
        await reload_weights(self.admin_clients)

        # Clean rollout directories at the beginning
        if self.step == 0:
            shutil.rmtree(
                get_rollout_dir(self.config.output_dir), ignore_errors=True
            )

    async def sync_policy(self):
        """
        Check for new weights from the Trainer and update Inference.

        Semantics mirror prime_rl.orchestrator.scheduler.update_policy:
        - Keep async_level = step - ckpt_step <= max_async_level
        - If strict_async_level:
            always use policy at (step - max_async_level) and wait if needed
        - Else:
            use the newest checkpoint, but never violate max_async_level
        """
        broadcast_dir = get_broadcast_dir(self.config.output_dir)

        # Latest checkpoint step that is actually present on disk
        latest_ckpt_step = get_latest_ckpt_step(broadcast_dir) or 0

        # Minimum checkpoint we are allowed to be on (enforce async bound)
        async_away_ckpt_step = max(self.step - self.config.max_async_level, 0)

        if self.config.strict_async_level:
            # Always lag exactly max_async_level behind (or 0 at the beginning)
            target_step = async_away_ckpt_step
        else:
            # Use the latest available checkpoint, but never violate async bound
            target_step = max(async_away_ckpt_step, latest_ckpt_step)

        if target_step <= self.ckpt_step:
            # Already at or ahead of the desired policy, nothing to do
            return

        # If we are forcing an async barrier, log explicitly
        if target_step == async_away_ckpt_step:
            self.logger.info(
                f"Hit async barrier: step={self.step}, "
                f"ckpt_step={self.ckpt_step}, "
                f"max_async_level={self.config.max_async_level}. "
                f"Waiting for checkpoint {target_step}."
            )

        # Wait for STABLE flag to appear for the target checkpoint
        step_dir = get_step_path(broadcast_dir, target_step)
        stable_path = step_dir / "STABLE"
        while not stable_path.exists():
            await asyncio.sleep(0.1)

        self.logger.info(f"Updating inference weights to step {target_step}")

        await update_weights(
            self.admin_clients,
            step_dir,
            lora_name=self.config.lora_name,
        )

        self.ckpt_step = target_step

    async def run_step(self):
        """Generate one batch of data using Ludic via BatchSource."""
        self.logger.info(f"Starting Step {self.step}")

        # 1. Sync policy with trainer checkpoints
        await self.sync_policy()

        # 2. Fetch next batch from the BatchSource
        saw_batch = await self.batch_source.next_batch()

        self.logger.info(
            f"Step {self.step}: Generated {len(saw_batch.items)} items. "
            f"Avg Reward: {saw_batch.meta.get('avg_total_reward', 0.0):.2f}"
        )

        # 3. Save for Trainer
        self.sink.save_step(saw_batch, self.step)

        self.step += 1

    async def loop(self):
        await self.setup()
        while self.step < (self.config.max_steps or float("inf")):
            await self.run_step()

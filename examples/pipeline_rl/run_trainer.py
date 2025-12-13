import logging
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# Rich Dashboard
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

# Ludic Imports
from ludic.inference.vllm_client import VLLMChatClient
from ludic.training.batching import PipelineBatchSource
from ludic.training.algorithm import make_reinforce
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.distributed.adapters import create_vllm_publisher

# ---------------------------------------------------------------------------
# Configuration (Must match Actor!)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 
REDIS_URL = "redis://localhost:6379/0"
QUEUE_KEY = "ludic_tictactoe_queue"

logging.basicConfig(level=logging.INFO)
console = Console()

def create_dashboard(stats: dict, step: int) -> Table:
    """Live metrics table."""
    table = Table(box=box.ROUNDED, title=f"🧠 Ludic Trainer (Step {step})", width=80)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    # RL Stats
    table.add_row("💰 Avg Reward", f"{stats.get('avg_total_reward', 0.0):+.4f}")
    table.add_row("📉 Loss", f"{stats.get('loss', 0.0):.4f}")
    
    # Errors
    syn_err = stats.get('err_syntax', 0.0)
    sem_err = stats.get('err_semantic', 0.0)
    table.add_row("🚫 Syntax Errors", f"{syn_err:.1%}")
    table.add_row("⚠️ Illegal Moves", f"{sem_err:.1%}")
    
    # Throughput
    table.add_row("📦 Batch Size", f"{stats.get('num_rollouts', 0)} rollouts / {stats.get('num_samples', 0)} samples")
    return table

def main():
    console.rule("[bold magenta]🔥 Starting TRAINER Node[/]")

    # 1. Setup vLLM Client for WEIGHT PUSHING
    # enable_weight_updates=True initializes the NCCL communicator
    client = VLLMChatClient(
        host=VLLM_HOST, 
        port=VLLM_PORT, 
        group_port=VLLM_GROUP_PORT, 
        enable_weight_updates=True,
        device="cuda:0"
    )
    publisher = create_vllm_publisher(client)

    # 2. Load Heavy Model & LoRA
    print(f"📥 Loading Model: {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0"
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 3. Setup Pipeline Source (Reads from Redis)
    batch_source = PipelineBatchSource(
        redis_url=REDIS_URL,
        queue_key=QUEUE_KEY,
        batch_size=16,
        poll_timeout=1
    )

    # 4. Setup Trainer
    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=TrainerConfig(
            model_device="cuda:0",
            lr=1e-4,
            grad_accum_steps=1,
            sync_every_steps=1,
            max_lag=2
        )
    )

    # 5. Training Loop
    with Live(console=console, refresh_per_second=4) as live:
        step = 0
        while True:
            step += 1
            # This BLOCKS until Actors push enough data to Redis
            stats = trainer.train_step_sync()
            live.update(create_dashboard(stats, step))

if __name__ == "__main__":
    main()

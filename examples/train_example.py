import sys
import time
import logging
import json
import requests
import torch
from transformers import AutoModelForCausalLM

# PEFT Imports
from peft import get_peft_model, LoraConfig, TaskType

# Rich Imports for Dashboard
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.training.rollout_engine import RolloutEngine, RolloutBatchSource
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the 7B Instruct model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 

# Training Hyperparameters
LEARNING_RATE = 1e-4  
NUM_TRAIN_STEPS = 50
BATCH_SIZE = 4       
MAX_STEPS_PER_EPISODE = 5 

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("lora_7b_trainer")

# ---------------------------------------------------------------------------
# Beautiful CLI Dashboard
# ---------------------------------------------------------------------------

def create_dashboard(stats: dict, step: int, total_steps: int) -> Table:
    """Creates a rich table for the current training step."""
    table = Table(box=box.ROUNDED, title=f"🚀 Ludic Training (Step {step}/{total_steps})", width=100)
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Visual", style="green")

    # 1. Main RL Stats
    reward = stats.get('avg_total_reward', 0.0)
    loss = stats.get('loss', 0.0)
    
    # Reward Bar (-1 to 1 range usually)
    r_bar_len = int((reward + 1.0) * 10) # map -1..1 to 0..20
    r_bar = "█" * max(0, r_bar_len)
    
    table.add_row("💰 Avg Reward", f"{reward:+.4f}", r_bar)
    table.add_row("📉 Loss", f"{loss:.4f}", "")
    
    table.add_section()

    # 2. Error Rates (The important part for you!)
    syn_err = stats.get('err_syntax', 0.0)
    sem_err = stats.get('err_semantic', 0.0)
    
    syn_color = "red" if syn_err > 0.1 else "yellow" if syn_err > 0.0 else "dim"
    sem_color = "red" if sem_err > 0.1 else "yellow" if sem_err > 0.0 else "dim"

    table.add_row("🚫 Syntax Errors", f"[{syn_color}]{syn_err:.1%}[/]", "Invalid XML (<move>...)")
    table.add_row("⚠️ Illegal Moves", f"[{sem_color}]{sem_err:.1%}[/]", "Occupied/OOB cell")

    table.add_section()

    # 3. Outcomes
    win = stats.get('rate_win', 0.0)
    loss_rate = stats.get('rate_loss', 0.0)
    draw = stats.get('rate_draw', 0.0)
    
    table.add_row("🏆 Win Rate", f"{win:.1%}", "")
    table.add_row("💀 Loss Rate", f"{loss_rate:.1%}", "")
    table.add_row("🤝 Draw Rate", f"{draw:.1%}", "")

    # 4. Tech Stats
    items = int(stats.get('batch_items', 0))
    bs = int(stats.get('batch_size', 0))
    table.add_section()
    table.add_row("📦 Batch Size", f"{bs} eps / {items} steps", "")

    return table

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def wait_for_server(url: str, console: Console, timeout_s: int = 600):
    """
    Waits for vLLM to be healthy. Raises TimeoutError if it takes too long.
    """
    console.print(f"⏳ Waiting for vLLM at {url} (timeout={timeout_s}s)...")
    logger.info(f"Connecting to vLLM at {url}")
    
    start_time = time.time()
    
    while True:
        # Check for timeout
        if time.time() - start_time > timeout_s:
            logger.error(f"Timeout waiting for vLLM at {url}")
            raise TimeoutError(f"vLLM server at {url} did not respond within {timeout_s} seconds.")

        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                console.print("✅ vLLM Server is online.")
                logger.info("vLLM Server online")
                return
        except requests.RequestException:
            pass
        
        time.sleep(2)

def main():
    console = Console()
    if not torch.cuda.is_available():
        sys.exit(1)
        
    print(f"🛡️ LoRA Trainer running on: {torch.cuda.get_device_name(0)}")

    # 1. Wait for vLLM (Ensure you ran it WITHOUT --enable-lora)
    wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}", console)

    # 2. Setup Client
    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        group_port=VLLM_GROUP_PORT,
        enable_weight_updates=True,
        device="cuda:0" 
    )
    publisher = create_vllm_publisher(client)

    # 3. Load Base Model (7B)
    print(f"🧠 Loading base model: {MODEL_NAME}...")
    # NOTE: If you run out of VRAM here, install `bitsandbytes` and add `load_in_4bit=True`
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto" # or "cuda:0" if single GPU
    )

    # 4. Apply LoRA Configuration
    print("💉 Injecting LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
        r=16,           # Rank: 16 is a good balance for 7B
        lora_alpha=32,  # Alpha usually 2x Rank
        lora_dropout=0.05,
        bias="none",
        # Target all linear projection layers for best performance
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(base_model, peft_config)
    
    # Print stats to confirm we are training only ~0.05% of params
    model.print_trainable_parameters() 

    # 5. Setup Engine
    env_registry = {"tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)}
    
    # --- PROTOCOL FACTORY ---
    # Custom factory to accept 'system_prompt' from ProtocolSpec.kwargs
    # and pass it to SingleAgentSyncProtocol's new 'prompt' argument.
    def create_protocol(system_prompt: str = None):
        return SingleAgentSyncProtocol(
            agent=Agent(client=client, model=MODEL_NAME, ctx=FullDialog(), parser=xml_move_parser),
            prompt=system_prompt  # <--- INJECTED HERE
        )

    protocol_registry = {
        "single_agent": create_protocol
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path="lora_7b_training.jsonl"
    )

    # 6. Setup Batch Source
    def make_requests():
        # Get the standard prompt from the env class just to append to it
        base_prompt = TicTacToeEnv().suggested_sysprompt or ""
        
        # Create our Training Prompt (XML Instructions)
        training_prompt = base_prompt + "\n\nOutput your move as a single XML tag, e.g., <move>A1</move>."

        sampling_args = {
            "temperature": 1.0, 
            "max_tokens": 100,
            "extras": {"extra_body": {"return_token_ids": True}} 
        }
        return [RolloutRequest(
            env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
            protocol=ProtocolSpec(kind="single_agent", kwargs={"system_prompt": training_prompt}),
            sampling_args=sampling_args, 
            num_episodes=BATCH_SIZE,
        )]

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=make_reinforce(gamma=0.99).credit_assigner,
        requests_fn=make_requests,
        max_steps=MAX_STEPS_PER_EPISODE,
        concurrency=BATCH_SIZE,
        retokenize=False 
    )

    # 7. Trainer
    # Uses the updated 'Trainer' which detects PEFT models and handles merging.
    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=TrainerConfig(
            model_device="cuda:0", 
            lr=LEARNING_RATE, 
            grad_accum_steps=4,  # Increase accum steps for 7B to stabilize gradients
            sync_every_steps=1
        )
    )

    # 8. Run with Live Dashboard
    print("🏋️ Starting LoRA Training on 7B Model...")
    
    with Live(console=console, refresh_per_second=4) as live:
        for i in range(1, NUM_TRAIN_STEPS + 1):
            # 1. Run step
            stats = trainer.train_step_sync()
            
            # 2. Update Dashboard
            dashboard = create_dashboard(stats, i, NUM_TRAIN_STEPS)
            live.update(dashboard)
            
            # 3. (Optional) Print raw json line to file for history
            with open("training_log.jsonl", "a") as f:
                f.write(json.dumps(stats) + "\n")

    console.print("[bold green]🎉 Training Done![/]")

if __name__ == "__main__":
    main()
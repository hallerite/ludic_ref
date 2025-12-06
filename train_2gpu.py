import os
import sys
import time
import logging
import requests
import torch
from transformers import AutoModelForCausalLM

# PEFT Imports
from peft import get_peft_model, LoraConfig, TaskType

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.training.rollout_engine import RolloutEngine, RolloutBatchSource, EnvRegistry, ProtocolRegistry
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.training.trainer import Trainer  # Uses the updated Trainer
from ludic.training.config import TrainerConfig
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher

# Import Env
try:
    from examples.envs.tic_tac_toe import TicTacToeEnv
except ImportError:
    print("⚠️ Could not import TicTacToeEnv from examples. Using Mock.")
    from tests._mocks import MockEnv as TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the 7B Instruct model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 

# Training Hyperparameters
# LoRA allows slightly higher LRs than full finetuning.
LEARNING_RATE = 1e-4  
NUM_TRAIN_STEPS = 50
BATCH_SIZE = 4       
MAX_STEPS_PER_EPISODE = 5 

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("lora_7b_trainer")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def wait_for_server(url: str):
    logger.info(f"⏳ Waiting for vLLM at {url}...")
    while True:
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                logger.info("✅ vLLM Server is online.")
                return
        except requests.RequestException:
            pass
        time.sleep(2)

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
        
    logger.info(f"🛡️ LoRA Trainer running on: {torch.cuda.get_device_name(0)}")

    # 1. Wait for vLLM (Ensure you ran it WITHOUT --enable-lora)
    wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}")

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
    logger.info(f"🧠 Loading base model: {MODEL_NAME}...")
    # NOTE: If you run out of VRAM here, install `bitsandbytes` and add `load_in_4bit=True`
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto" # or "cuda:0" if single GPU
    )

    # 4. Apply LoRA Configuration
    logger.info("💉 Injecting LoRA adapters...")
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
    protocol_registry = {
        "single_agent": lambda: SingleAgentSyncProtocol(
            agent=Agent(client=client, model=MODEL_NAME, ctx=FullDialog(), parser=xml_move_parser)
        )
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path="lora_7b_training.jsonl"
    )

    # 6. Setup Batch Source
    def make_requests():
        sys_prompt = "You are playing Tic-Tac-Toe. Output your move as a single XML tag, e.g., <move>A1</move>."
        sampling_args = {
            "temperature": 1.0, 
            "max_tokens": 100,
            "extras": {"extra_body": {"return_token_ids": True}} 
        }
        return [RolloutRequest(
            env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
            protocol=ProtocolSpec(kind="single_agent"),
            sampling_args=sampling_args, 
            num_episodes=BATCH_SIZE,
            meta={"system_prompt": sys_prompt}
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
    # We use the updated 'Trainer' class which automatically detects
    # the PEFT model and handles the Merge -> Sync -> Unmerge workflow.
    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        publisher=publisher,
        cfg=TrainerConfig(
            model_device="cuda:0", 
            lr=LEARNING_RATE, 
            grad_accum_steps=4,  # Increase accum steps for 7B to stabilize gradients
            sync_every_steps=1
        )
    )

    # 8. Run
    logger.info("🏋️ Starting LoRA Training on 7B Model...")
    trainer.train_sync(
        num_steps=NUM_TRAIN_STEPS,
        log_every=1,
        log_fn=lambda stats: logger.info(
            f"Step {stats['train_step']:.0f} | Loss: {stats['loss']:.4f} | Reward: {stats['avg_total_reward']:.2f}"
        )
    )
    logger.info("🎉 Done.")

if __name__ == "__main__":
    main()
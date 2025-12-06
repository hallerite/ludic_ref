import os
import sys
import time
import logging
import requests
import torch
from transformers import AutoModelForCausalLM

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.training.rollout_engine import RolloutEngine, RolloutBatchSource, EnvRegistry, ProtocolRegistry
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher

# Import Env
try:
    from examples.envs.tic_tac_toe import TicTacToeEnv
except ImportError:
    # Fallback if running from a different directory
    print("⚠️ Could not import TicTacToeEnv from examples. Using Mock.")
    from tests._mocks import MockEnv as TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 

# Training Hyperparameters
NUM_TRAIN_STEPS = 50
BATCH_SIZE = 8       
MAX_STEPS_PER_EPISODE = 9 

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("trainer")

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
    # 1. Device Setup
    # We rely on the user setting CUDA_VISIBLE_DEVICES externally.
    # So "cuda:0" here refers to whatever GPU this process is allowed to see.
    if not torch.cuda.is_available():
        logger.error("❌ No GPU found! Did you set CUDA_VISIBLE_DEVICES correctly?")
        sys.exit(1)
        
    logger.info(f"🛡️ Trainer running on: {torch.cuda.get_device_name(0)}")

    # 2. Wait for vLLM (launched externally)
    wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}")

    # 3. Setup Client
    logger.info("🔗 Connecting Client...")
    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        group_port=VLLM_GROUP_PORT,
        enable_weight_updates=True,
        device="cuda:0" 
    )
    publisher = create_vllm_publisher(client)

    # 4. Load Model
    logger.info("🧠 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to("cuda:0")

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
        jsonl_path="training_rollouts.jsonl"
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
    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        publisher=publisher,
        cfg=TrainerConfig(model_device="cuda:0", lr=1e-6, grad_accum_steps=1, sync_every_steps=1)
    )

    # 8. Run
    logger.info("🏋️ Starting Training...")
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
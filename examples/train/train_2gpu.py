import os
import sys
import time
import subprocess
import logging
import requests
import torch
from transformers import AutoModelForCausalLM

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.training.rollout_engine import (
    RolloutEngine,
    RolloutBatchSource,
    EnvRegistry,
    ProtocolRegistry,
)
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher

# Import the example Env
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 

# PHYSICAL GPU IDs
# GPU 0 -> vLLM Inference
# GPU 1 -> Trainer/Gradient Calculation
PHYSICAL_SERVER_GPU = "0"
PHYSICAL_TRAINER_GPU = "1"

# Training Hyperparameters
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 8       
MAX_STEPS_PER_EPISODE = 9 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer_script")

# ---------------------------------------------------------------------------
# Helper: Manage vLLM Server Process
# ---------------------------------------------------------------------------

def launch_vllm_server():
    """Launches vLLM strictly on Physical GPU 0."""
    env = os.environ.copy()
    
    # ISOLATION: Restrict vLLM to Physical GPU 0
    env["CUDA_VISIBLE_DEVICES"] = PHYSICAL_SERVER_GPU
    # Required for the V1 engine (standard in Ludic)
    env["VLLM_USE_V1"] = "1" 
    
    cmd = [
        sys.executable, "-m", "ludic.inference.vllm_server",
        "--model", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.6", 
        "--max-model-len", "2048",
        "--enforce-eager",
        # Enable the weight sync extension
        "--enable-lora", "False" 
    ]
    
    logger.info(f"🚀 Launching vLLM on Physical GPU {PHYSICAL_SERVER_GPU}...")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_server(url: str, timeout: int = 120):
    start = time.time()
    logger.info("⏳ Waiting for vLLM health check...")
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                logger.info("✅ vLLM Server is ready.")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False

# ---------------------------------------------------------------------------
# Main Training Logic
# ---------------------------------------------------------------------------

def main():
    # 1. Start vLLM Server (It will isolate itself to GPU 0)
    server_proc = launch_vllm_server()
    
    try:
        # ISOLATION: Mask GPU 0 from the Trainer process.
        # From now on, Physical GPU 1 is seen as "cuda:0" by PyTorch in this process.
        os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_TRAINER_GPU
        
        # Verify isolation
        if torch.cuda.is_available():
            logger.info("🛡️ Trainer Process Isolation Active.")
            logger.info(f"   Visible Device 0 is physically: {PHYSICAL_TRAINER_GPU}")
        
        if not wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}"):
            raise RuntimeError("vLLM server failed to start.")

        # 2. Setup Client & Publisher
        # The client connects to vLLM (HTTP) and establishes the NCCL group (TCP).
        logger.info("🔗 Connecting Client...")
        client = VLLMChatClient(
            host=VLLM_HOST,
            port=VLLM_PORT,
            group_port=VLLM_GROUP_PORT,
            enable_weight_updates=True,
            device="cuda:0" # This is Physical GPU 1
        )
        
        # The publisher handles sending weights from Trainer -> vLLM
        publisher = create_vllm_publisher(client)

        # 3. Load Model for Training
        logger.info("🧠 Loading model for training...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        # We do not use FSDP here, just a standard model moved to GPU
        model = model.to("cuda:0")
        
        # 4. Configure Rollout Engine
        env_registry: EnvRegistry = {
            "tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)
        }
        
        # We need a fresh agent for every rollout task
        def create_protocol():
            return SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=MODEL_NAME, 
                    ctx=FullDialog(),
                    parser=xml_move_parser
                )
            )

        protocol_registry: ProtocolRegistry = {
            "single_agent": create_protocol
        }

        engine = RolloutEngine(
            env_registry=env_registry,
            protocol_registry=protocol_registry,
            jsonl_path="training_rollouts.jsonl"
        )

        # 5. Configure Batch Source
        def make_requests() -> list[RolloutRequest]:
            sys_prompt = (
                "You are playing Tic-Tac-Toe. "
                "Output your move as a single XML tag, e.g., <move>A1</move>."
            )
            
            # Use 'extras' to tell vLLM to return token IDs.
            # This allows us to use retokenize=False (faster).
            sampling_args = {
                "temperature": 1.0,
                "max_tokens": 100,
                "extras": {"extra_body": {"return_token_ids": True}} 
            }

            req = RolloutRequest(
                env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
                protocol=ProtocolSpec(kind="single_agent"),
                sampling_args=sampling_args, 
                num_episodes=BATCH_SIZE,
                meta={"system_prompt": sys_prompt}
            )
            return [req]

        # Use REINFORCE
        algorithm = make_reinforce(gamma=0.99)

        batch_source = RolloutBatchSource(
            orchestrator=engine,
            credit_assigner=algorithm.credit_assigner,
            requests_fn=make_requests,
            max_steps=MAX_STEPS_PER_EPISODE,
            concurrency=BATCH_SIZE,
            retokenize=False 
        )

        # 6. Setup Trainer
        trainer_cfg = TrainerConfig(
            model_device="cuda:0",
            lr=1e-6,
            grad_accum_steps=1,
            sync_every_steps=1, # Sync weights to vLLM every step
            max_grad_norm=1.0
        )

        logger.info("🏋️ Starting Training Loop...")
        trainer = Trainer(
            model=model,
            algo=algorithm,
            batch_source=batch_source,
            publisher=publisher,
            cfg=trainer_cfg
        )

        # 7. Run Synchronous Training
        # We pass a simple logging lambda
        trainer.train_sync(
            num_steps=NUM_TRAIN_STEPS,
            log_every=1,
            log_fn=lambda stats: logger.info(
                f"Step {stats['train_step']:.0f} | "
                f"Loss: {stats['loss']:.4f} | "
                f"Reward: {stats['avg_total_reward']:.2f}"
            )
        )
        
        logger.info("🎉 Training Complete!")

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}", exc_info=True)
    finally:
        if server_proc:
            logger.info("🛑 Shutting down vLLM server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

if __name__ == "__main__":
    main()
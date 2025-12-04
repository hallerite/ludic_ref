import os
import sys
import time
import subprocess
import logging
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

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
from ludic.training.hf_trainer import LudicHFTrainer
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 

# PHYSICAL GPU IDs (for isolation)
PHYSICAL_SERVER_GPU = "0"
PHYSICAL_TRAINER_GPU = "1"

# Training Hyperparameters
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 8       
GRAD_ACCUM = 1       
LEARNING_RATE = 1e-6
MAX_STEPS_PER_EPISODE = 9 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer_script")

# ---------------------------------------------------------------------------
# Helper: Manage vLLM Server Process
# ---------------------------------------------------------------------------

def launch_vllm_server():
    """Launches vLLM strictly on Physical GPU 0."""
    env = os.environ.copy()
    
    # ISOLATION STEP 1: Restrict vLLM to Physical GPU 0
    env["CUDA_VISIBLE_DEVICES"] = PHYSICAL_SERVER_GPU
    env["VLLM_USE_V1"] = "1" 
    
    cmd = [
        sys.executable, "-m", "ludic.inference.vllm_server",
        "--model", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        # Memory Safety: limit to 40% just in case, though isolation handles most risks
        "--gpu-memory-utilization", "0.4", 
        "--max-model-len", "2048",
        "--enforce-eager",
    ]
    
    logger.info(f"🚀 Launching vLLM on Physical GPU {PHYSICAL_SERVER_GPU}...")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_server(url: str, timeout: int = 120):
    start = time.time()
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
    # 1. Start vLLM Server (It will take GPU 0)
    server_proc = launch_vllm_server()
    
    try:
        # ISOLATION STEP 2: Mask GPU 0 from this process immediately.
        # Now, Physical GPU 1 becomes Logical "cuda:0" for this script.
        os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_TRAINER_GPU
        
        # Verify isolation
        if torch.cuda.is_available():
            logger.info("🛡️ Trainer Process Isolation Active.")
            logger.info(f"   Physical GPU {PHYSICAL_TRAINER_GPU} is mapped to Logical: {torch.cuda.get_device_name(0)}")
        
        # Wait for server *after* setting env vars to ensure we don't accidentally init cuda on 0
        if not wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}"):
            raise RuntimeError("vLLM server failed to start.")

        # 2. Setup Client
        # Note: device="cuda:0" here refers to the Trainer's ONLY visible GPU (Physical 1)
        logger.info(f"🔗 Connecting Client on Logical GPU 0 (Physical {PHYSICAL_TRAINER_GPU})...")
        client = VLLMChatClient(
            host=VLLM_HOST,
            port=VLLM_PORT,
            group_port=VLLM_GROUP_PORT,
            enable_weight_updates=True,
            device="cuda:0" 
        )

        # 3. Load Model
        logger.info("🧠 Loading model on Logical GPU 0...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).to("cuda:0") # Move to the only visible GPU
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 4. Configure Rollout Source
        env_registry: EnvRegistry = {
            "tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)
        }
        
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
            jsonl_path="rollouts.jsonl"
        )

        def make_requests() -> list[RolloutRequest]:
            sys_prompt = (
                "You are playing Tic-Tac-Toe. "
                "Output your move as a single XML tag, e.g., <move>A1</move>."
            )
            
            # FIXED: Sampling Args structure
            sampling_args = {
                "seed": 42,
                "temperature": 1.0,
                "max_tokens": 100,
                "stop": [],
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

        batch_source = RolloutBatchSource(
            orchestrator=engine,
            credit_assigner=make_reinforce(gamma=0.99).credit_assigner,
            requests_fn=make_requests,
            max_steps=MAX_STEPS_PER_EPISODE,
            concurrency=BATCH_SIZE,
            retokenize=False 
        )

        # 5. Configure RL Algorithm & Trainer
        algorithm = make_reinforce(gamma=0.99)

        hf_args = TrainingArguments(
            output_dir="./checkpoints",
            logging_steps=1,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE, 
            learning_rate=LEARNING_RATE,
            report_to=["none"],
            dataloader_num_workers=0, 
            remove_unused_columns=False,
            # HF Trainer will auto-detect "cuda:0" as the only device
            no_cuda=False 
        )

        logger.info("🏋️ Starting Training...")
        trainer = LudicHFTrainer(
            model=model,
            rl_algorithm=algorithm,
            batch_source=batch_source,
            client=client,
            args=hf_args,
            pad_token_id=tokenizer.pad_token_id,
        )

        trainer.train()
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
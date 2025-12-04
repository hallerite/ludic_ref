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
from ludic.training.types import (
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
)
from ludic.types import SamplingArgs
from ludic.training.algorithm import make_reinforce
from ludic.training.hf_trainer import LudicHFTrainer
from ludic.interaction.single_agent import SingleAgentSyncProtocol

# Import the example Env (ensure this is in your python path)
# If running from the root of the tree provided, this path should work
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 # For NCCL weight syncing

# GPU Assignment
SERVER_GPU_ID = "0"
TRAINER_GPU_ID = "1"

# Training Hyperparameters
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 8       # Rollouts per step
GRAD_ACCUM = 1       # Update every step (total batch size = 8)
LEARNING_RATE = 1e-6
MAX_STEPS_PER_EPISODE = 9 # Max moves in TicTacToe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer_script")

# ---------------------------------------------------------------------------
# Helper: Manage vLLM Server Process
# ---------------------------------------------------------------------------

def launch_vllm_server():
    """Launches vLLM on SERVER_GPU_ID."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = SERVER_GPU_ID
    env["VLLM_USE_V1"] = "1" 
    
    cmd = [
        sys.executable, "-m", "ludic.inference.vllm_server",
        "--model", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "2048",
        "--enforce-eager", # Often needed for smaller batches/RL loops
    ]
    
    logger.info(f"🚀 Launching vLLM on GPU {SERVER_GPU_ID}...")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_server(url: str, timeout: int = 120):
    """Polls /health until server is ready."""
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
    # 1. Start vLLM Server
    server_proc = launch_vllm_server()
    
    try:
        if not wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}"):
            raise RuntimeError("vLLM server failed to start.")

        # 2. Setup Client (The Bridge)
        # We bind the client's NCCL communicator to the TRAINER GPU so it can 
        # push weights from the training model to the server.
        logger.info(f"🔗 Connecting Client (Weight Updates Enabled) on GPU {TRAINER_GPU_ID}...")
        client = VLLMChatClient(
            host=VLLM_HOST,
            port=VLLM_PORT,
            group_port=VLLM_GROUP_PORT,
            enable_weight_updates=True,
            device=f"cuda:{TRAINER_GPU_ID}" 
        )

        # 3. Load Trainable Model (HF) on Trainer GPU
        logger.info(f"🧠 Loading trainable model on GPU {TRAINER_GPU_ID}...")
        # Note: We load the model manually to ensure it goes to the right device immediately
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        
        # We need the tokenizer for the DataCollator (padding)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 4. Configure Rollout Source (Data Pipeline)
        
        # Define the registries
        env_registry: EnvRegistry = {
            "tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)
        }
        
        # Helper to build the protocol (Agent + Context + Parser)
        def create_protocol():
            return SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=MODEL_NAME, # Agent uses the vLLM model name
                    ctx=FullDialog(),
                    parser=xml_move_parser
                )
            )

        protocol_registry: ProtocolRegistry = {
            "single_agent": create_protocol
        }

        # The Engine runs the episodes
        engine = RolloutEngine(
            env_registry=env_registry,
            protocol_registry=protocol_registry,
            jsonl_path="rollouts.jsonl" # Optional: log games to file
        )

        # Function to generate requests for the batch source
        # We ask vLLM to return token_ids so we don't need to re-tokenize manually
        def make_requests() -> list[RolloutRequest]:
            # TicTacToe System Prompt + XML instruction
            sys_prompt = (
                "You are playing Tic-Tac-Toe. "
                "Output your move as a single XML tag, e.g., <move>A1</move>."
            )
            
            # FIXED: Define sampling_args as a dict with the 'extras' key preserved.
            # Do NOT use .to_openai_kwargs() here.
            sampling_args: SamplingArgs = {
                "seed": 42,
                "temperature": 1.0,
                "max_tokens": 100,
                "stop": [],
                # The agent looks for this 'extras' key explicitly
                "extras": {"extra_body": {"return_token_ids": True}} 
            }

            req = RolloutRequest(
                env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
                protocol=ProtocolSpec(kind="single_agent"),
                sampling_args=sampling_args, # Pass the dict directly
                num_episodes=BATCH_SIZE,
                meta={"system_prompt": sys_prompt}
            )
            return [req]

        # The Batch Source feeds the Trainer
        # We use make_reinforce (Standard PG) -> MonteCarloReturn
        # We do NOT enable retokenize, because vLLM gives us the IDs.
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

        # HF Training Arguments (logging, saving, etc.)
        hf_args = TrainingArguments(
            output_dir="./checkpoints",
            logging_steps=1,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE, # handled by our collation
            learning_rate=LEARNING_RATE,
            report_to=["none"],
            dataloader_num_workers=0, # Required by LudicHFTrainer
            remove_unused_columns=False,
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

        # 6. Train!
        trainer.train()
        
        logger.info("🎉 Training Complete!")

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}", exc_info=True)
    finally:
        # Cleanup: Terminate vLLM server
        if server_proc:
            logger.info("🛑 Shutting down vLLM server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

if __name__ == "__main__":
    main()
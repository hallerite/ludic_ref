import os
import sys
import time
import signal
import subprocess
import asyncio
import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.sampling import SamplingArgs
from ludic.parsers import xml_move_parser
from ludic.training.algorithm import make_reinforce
from ludic.training.config import TrainerConfig
from ludic.training.rollout_engine import (
    RolloutEngine,
    RolloutBatchSource,
    EnvRegistry,
    ProtocolRegistry,
)
from ludic.training.trainer import Trainer
from ludic.training.types import (
    RolloutRequest,
    EnvSpec,
    ProtocolSpec,
)
from ludic.interaction.single_agent import SingleAgentSyncProtocol

from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_PORT = 8000
NCCL_PORT = 51216

# Training Params
BATCH_SIZE = 8       # Episodes per step
NUM_STEPS = 10       # Total training steps
LR = 1e-5

# ---------------------------------------------------------------------------
# Helper: Launch vLLM on GPU 0
# ---------------------------------------------------------------------------

def launch_vllm_server_on_gpu0():
    """
    Launches vLLM as a subprocess.
    Crucially, we override CUDA_VISIBLE_DEVICES to '0' for this process,
    even though the main script is running on '1'.
    """
    print(f"🚀 Launching vLLM server on GPU 0 (Port {VLLM_PORT})...")
    
    # Copy current env, but force GPU 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    cmd = [
        sys.executable, "-m", "ludic.inference.vllm_server",
        "--model", MODEL_NAME,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "2048",
        "--enforce-eager",             # Often more stable for RL loops
        "--disable-log-stats",         # Reduce noise
    ]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,     # Redirect stdout to keep main logs clean
        stderr=subprocess.PIPE,        # Capture stderr for debugging if needed
        text=True
    )
    return process

def wait_for_vllm(process):
    """Polls the vLLM health endpoint until it is up."""
    import requests
    url = f"http://127.0.0.1:{VLLM_PORT}/health"
    print("⏳ Waiting for vLLM to be healthy...")
    
    for _ in range(60): # Wait up to 60s
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"vLLM process died!\nSTDERR: {stderr}")
        try:
            if requests.get(url, timeout=1).status_code == 200:
                print("✅ vLLM is ready!")
                return
        except Exception:
            time.sleep(1)
            
    raise TimeoutError("vLLM failed to start in time.")

# ---------------------------------------------------------------------------
# Main Training Script
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Start the vLLM Server (GPU 0)
    vllm_proc = launch_vllm_server_on_gpu0()
    
    try:
        wait_for_vllm(vllm_proc)

        # 2. Setup Local Components (GPU 1)
        # The main script runs on GPU 1 (provided by the user call),
        # so we load the student model here.
        print(f"📥 Loading local student model: {MODEL_NAME} on {os.getenv('CUDA_VISIBLE_DEVICES', 'default')}...")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model for training
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).cuda() # Moves to GPU 1 automatically based on CUDA_VISIBLE_DEVICES

        # 3. Initialize Ludic Client
        # enable_weight_updates=True is CRITICAL. It tells the client to act
        # as a rank in the NCCL group to push weights to vLLM.
        client = VLLMChatClient(
            host="127.0.0.1",
            port=VLLM_PORT,
            group_port=NCCL_PORT,
            enable_weight_updates=True 
        )

        # 4. Setup Registries
        # Define how to build the Env and the Protocol
        env_registry: EnvRegistry = {
            "tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)
        }
        
        # We need a protocol factory that creates an agent wired to our shared client
        def create_protocol(system_prompt: str):
            return SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=MODEL_NAME,
                    ctx=FullDialog(system_prompt=system_prompt),
                    parser=xml_move_parser
                )
            )

        protocol_registry: ProtocolRegistry = {
            "single_agent": create_protocol
        }

        # 5. Setup Rollout Engine & Request Factory
        engine = RolloutEngine(
            env_registry=env_registry,
            protocol_registry=protocol_registry,
            jsonl_path="rollouts.jsonl"
        )

        # Helper to generate the System Prompt with XML instructions
        def get_system_prompt():
            base = TicTacToeEnv().suggested_sysprompt
            return base + "\nReply with a single XML tag, e.g. <move>A1</move>."

        # Define the Batch Source
        # This function generates the list of requests for every training step
        def make_requests() -> List[RolloutRequest]:
            return [
                RolloutRequest(
                    env=EnvSpec(kind="tictactoe"),
                    protocol=ProtocolSpec(
                        kind="single_agent", 
                        kwargs={"system_prompt": get_system_prompt()}
                    ),
                    num_episodes=BATCH_SIZE,
                    sampling_args=SamplingArgs(
                        temperature=0.8, 
                        max_tokens=100
                    )
                )
            ]

        # REINFORCE typically benefits from re-tokenizing locally to ensure
        # gradients align perfectly with the training model's embeddings.
        batch_source = RolloutBatchSource(
            orchestrator=engine,
            credit_assigner=make_reinforce(gamma=1.0).credit_assigner,
            requests_fn=make_requests,
            max_steps=10,
            concurrency=8,
            retokenize=True,
            tokenize=lambda text: tokenizer.encode(text, add_special_tokens=False)
        )

        # 6. Setup Trainer
        # Configures the optimization loop
        trainer_config = TrainerConfig(
            lr=LR,
            model_device="cuda",  # Refers to the local device (GPU 1)
            max_grad_norm=1.0,
            sync_every_steps=1,   # Push weights to vLLM every step
        )

        trainer = Trainer(
            model=model,
            algo=make_reinforce(gamma=1.0),
            batch_source=batch_source,
            client=client,
            cfg=trainer_config
        )

        # 7. Run Training
        print("\n🥊 Starting Training Loop...")
        print(f"   Device 0: vLLM Server (Inference)")
        print(f"   Device 1: Trainer (Backprop)")
        
        def log_progress(stats):
            print(f"[Step {int(stats['train_step'])}] "
                  f"Reward: {stats['avg_total_reward']:.2f} | "
                  f"Loss: {stats['loss']:.4f}")

        # Use the synchronous wrapper for the script
        trainer.train_sync(
            num_steps=NUM_STEPS,
            log_every=1,
            log_fn=log_progress
        )

        print("✅ Training complete.")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup vLLM process
        if vllm_proc:
            print("💀 Killing vLLM server...")
            os.kill(vllm_proc.pid, signal.SIGTERM)

if __name__ == "__main__":
    main()
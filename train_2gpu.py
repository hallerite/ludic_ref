import os
import sys
import time
import subprocess
import logging
import multiprocessing
import requests

# NOTE: We keep top-level imports minimal to avoid premature CUDA init in the main process
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216 

# PHYSICAL GPU IDs
PHYSICAL_SERVER_GPU = "0"   # vLLM
PHYSICAL_TRAINER_GPU = "1"  # Trainer

# Hyperparams
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 8       
MAX_STEPS_PER_EPISODE = 9 

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Process 1: vLLM Server
# ---------------------------------------------------------------------------

def launch_vllm_server():
    """Launches vLLM as a subprocess strictly on Physical GPU 0."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = PHYSICAL_SERVER_GPU
    env["VLLM_USE_V1"] = "1" 
    
    cmd = [
        sys.executable, "-m", "ludic.inference.vllm_server",
        "--model", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.8", # vLLM gets 80% of GPU 0
        "--max-model-len", "2048",
        "--enforce-eager",
        "--enable-lora", "False" 
    ]
    
    logger.info(f"🚀 [Main] Launching vLLM on Physical GPU {PHYSICAL_SERVER_GPU}...")
    # Redirect stdout/stderr to avoid clutter, or let them flow
    process = subprocess.Popen(cmd, env=env)
    return process

def wait_for_server(url: str, timeout: int = 120):
    start = time.time()
    logger.info("⏳ [Main] Waiting for vLLM health check...")
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                logger.info("✅ [Main] vLLM Server is ready.")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False

# ---------------------------------------------------------------------------
# Process 2: The Trainer
# ---------------------------------------------------------------------------

def run_trainer_process():
    """
    This function runs in a completely separate process.
    We can set CUDA_VISIBLE_DEVICES here safely before importing torch.
    """
    # 1. FORCE ISOLATION IMMEDIATELY
    os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_TRAINER_GPU
    
    # 2. Late imports to ensure they see the restricted GPU environment
    import torch
    from transformers import AutoModelForCausalLM
    from ludic.agent import Agent
    from ludic.context.full_dialog import FullDialog
    from ludic.inference.vllm_client import VLLMChatClient
    from ludic.parsers import xml_move_parser
    from ludic.training.rollout_engine import RolloutEngine, RolloutBatchSource, EnvRegistry, ProtocolRegistry
    from ludic.training.algorithm import make_reinforce
    from ludic.training.trainer import Trainer
    from ludic.training.config import TrainerConfig
    from ludic.interaction.single_agent import SingleAgentSyncProtocol
    from ludic.distributed.adapters import create_vllm_publisher
    from examples.envs.tic_tac_toe import TicTacToeEnv

    # Setup Logging for this process
    t_logger = logging.getLogger("trainer")
    t_logger.setLevel(logging.INFO)

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        t_logger.info(f"🛡️ [Trainer] Process started. Visible GPU: {torch.cuda.get_device_name(current_device)}")
        t_logger.info(f"   (This should match Physical GPU {PHYSICAL_TRAINER_GPU})")

    # 3. Setup Client (Logical cuda:0 here is actually Physical GPU 1)
    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        group_port=VLLM_GROUP_PORT,
        enable_weight_updates=True,
        device="cuda:0" 
    )
    publisher = create_vllm_publisher(client)

    # 4. Load Model
    t_logger.info("🧠 [Trainer] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to("cuda:0")

    # 5. Setup Components
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

    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        publisher=publisher,
        cfg=TrainerConfig(model_device="cuda:0", lr=1e-6, grad_accum_steps=1, sync_every_steps=1)
    )

    # 6. Train
    t_logger.info("🏋️ [Trainer] Starting loop...")
    trainer.train_sync(
        num_steps=NUM_TRAIN_STEPS,
        log_every=1,
        log_fn=lambda stats: t_logger.info(
            f"Step {stats['train_step']:.0f} | Loss: {stats['loss']:.4f} | Reward: {stats['avg_total_reward']:.2f}"
        )
    )
    t_logger.info("🎉 [Trainer] Done!")

# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    # Ensure multiprocessing uses 'spawn' to get fresh processes without inherited state
    multiprocessing.set_start_method("spawn", force=True)

    # 1. Launch vLLM
    server_proc = launch_vllm_server()
    
    trainer_proc = None
    try:
        # 2. Wait for vLLM
        if not wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}"):
            raise RuntimeError("vLLM server failed to start.")

        # 3. Launch Trainer as a completely separate process
        logger.info("🚀 [Main] Spawning Trainer process on GPU 1...")
        trainer_proc = multiprocessing.Process(target=run_trainer_process)
        trainer_proc.start()
        
        # Wait for trainer to finish
        trainer_proc.join()
        
        if trainer_proc.exitcode != 0:
            logger.error("❌ Trainer process exited with error.")
        else:
            logger.info("✅ Trainer process finished successfully.")

    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user.")
    finally:
        if trainer_proc and trainer_proc.is_alive():
            trainer_proc.terminate()
        if server_proc:
            logger.info("🛑 Shutting down vLLM server...")
            server_proc.terminate()
            server_proc.wait()

if __name__ == "__main__":
    main()
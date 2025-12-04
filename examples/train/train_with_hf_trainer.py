import os
import sys
import time
import subprocess
import signal
import logging
import asyncio
from typing import Dict, Any

# Ensure we can import examples.envs
sys.path.append(os.getcwd())

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from ludic.inference.sampling import SamplingConfig
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# 1. Self-Contained NCCL Client (Hidden Shenanigans)
# ---------------------------------------------------------------------------
import requests
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

class MinimalVLLMClient:
    """
    A unified client that handles HTTP generation and NCCL weight syncing.
    """
    def __init__(self, host: str, port: int, group_port: int, accelerator: Accelerator):
        self.base_url = f"http://{host}:{port}"
        self.group_port = group_port
        self.accelerator = accelerator
        self.session = requests.Session()
        self.pynccl_comm = None
        self.rank = None

    def wait_for_server(self, timeout=60):
        print(f"⏳ Connecting to vLLM at {self.base_url}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                if self.session.get(f"{self.base_url}/health", timeout=1).status_code == 200:
                    print("✅ vLLM Connected")
                    return
            except:
                time.sleep(1)
        raise TimeoutError("vLLM failed to start")

    def init_sync(self):
        """Initializes the NCCL connection using Accelerator's device info."""
        # 1. Get world size from vLLM
        resp = self.session.get(f"{self.base_url}/get_world_size")
        vllm_world_size = resp.json()["world_size"]
        
        # 2. Configure our rank (we are the client, so we are rank + 1)
        total_world_size = vllm_world_size + 1
        self.rank = vllm_world_size

        # 3. Tell vLLM to open its side of the connection
        # Note: vLLM's host is 0.0.0.0 from its perspective
        self.session.post(f"{self.base_url}/init_communicator", json={
            "host": "0.0.0.0", "port": self.group_port, "world_size": total_world_size
        })
        time.sleep(0.5) # Allow vLLM to bind port

        # 4. Create our side using Accelerator's device
        pg = StatelessProcessGroup.create(
            host="127.0.0.1", port=self.group_port, rank=self.rank, world_size=total_world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.accelerator.device)

    def sync_weights(self, model):
        """Push all trainable weights to vLLM via NCCL."""
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            
            # 1. Notify vLLM via HTTP
            dtype = str(param.dtype)
            shape = tuple(param.shape)
            self.session.post(f"{self.base_url}/update_named_param", json={
                "name": name, "dtype": dtype, "shape": shape
            })
            
            # 2. Push Data via NCCL
            self.pynccl_comm.broadcast(param.data, src=self.rank)
            self.pynccl_comm.group.barrier()
        
        # Reset cache so vLLM uses new weights
        self.session.post(f"{self.base_url}/reset_prefix_cache")

    def generate(self, prompts: list[str], sampling_params: dict):
        """Standard OpenAI-compatible completion"""
        resp = self.session.post(f"{self.base_url}/v1/completions", json={
            "model": "model", # dummy name
            "prompt": prompts,
            **sampling_params
        })
        return resp.json()

# ---------------------------------------------------------------------------
# 2. The Training Script
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_PORT = 8000
NCCL_PORT = 51216

def main():
    # --- A. Initialize Accelerate ---
    # This handles device placement, un-wrapping, and distributed setup automatically.
    accelerator = Accelerator()
    
    # --- B. Launch vLLM (Background) ---
    # We force vLLM to GPU 0. Accelerator will put our script on GPU 1.
    if accelerator.is_main_process:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        vllm_cmd = [
            sys.executable, "-m", "ludic.inference.vllm_server",
            "--model", MODEL_NAME,
            "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.8",
            "--disable-log-stats"
        ]
        vllm_proc = subprocess.Popen(vllm_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    try:
        # --- C. Load Model & components ---
        print(f"Device: {accelerator.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        optimizer = AdamW(model.parameters(), lr=1e-5)

        # Accelerate prepares everything (moves to GPU, handles DDP if needed)
        model, optimizer = accelerator.prepare(model, optimizer)

        # --- D. Initialize Client & Sync ---
        client = MinimalVLLMClient("127.0.0.1", VLLM_PORT, NCCL_PORT, accelerator)
        client.wait_for_server()
        client.init_sync()

        # --- E. Training Loop (Simple REINFORCE) ---
        env = TicTacToeEnv()
        prompt_template = env.suggested_sysprompt + "\n<move>A1</move>\n"
        
        for step in range(10):
            # 1. Generation (Rollout)
            # We use a simplified loop here: 1 game per step for demonstration
            prompts = [prompt_template]
            
            # Ask vLLM to play
            output = client.generate(prompts, {"max_tokens": 10, "temperature": 0.8})
            text = output['choices'][0]['text']
            
            # Simple Reward Logic (Did we parse XML?)
            reward = 1.0 if "<move>" in text else -1.0
            
            # 2. Tokenization & Tensors
            inputs = tokenizer(prompts[0] + text, return_tensors="pt").to(accelerator.device)
            
            # 3. Forward Pass (on Student)
            outputs = model(**inputs)
            log_probs = outputs.logits.log_softmax(dim=-1)
            
            # 4. Loss (Vanilla Policy Gradient)
            # Only train on the generated tokens
            input_ids = inputs.input_ids[0]
            gen_start = len(tokenizer.encode(prompts[0]))
            
            action_log_probs = log_probs[0, gen_start-1:-1, :]
            action_tokens = input_ids[gen_start:]
            
            selected_log_probs = action_log_probs.gather(1, action_tokens.unsqueeze(-1))
            loss = -selected_log_probs.mean() * reward

            # 5. Backward & Step
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # 6. Sync Weights to vLLM
            print(f"Step {step+1}: Reward={reward} | Syncing weights...")
            client.sync_weights(model)

    finally:
        if accelerator.is_main_process:
            print("💀 Killing vLLM...")
            os.kill(vllm_proc.pid, signal.SIGTERM)

if __name__ == "__main__":
    main()
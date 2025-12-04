from __future__ import annotations

import asyncio
import threading
import logging
from collections import defaultdict
from typing import Dict, List, Optional, TypeVar, Coroutine, Any
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import IterableDataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ludic.inference.client import ChatClient
from ludic.training.algorithm import RLAlgorithm
from ludic.training.types import SAWItem, BatchSource

logger = logging.getLogger(__name__)

# Optional LoRA imports
try:
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    is_peft_available = True
except ImportError:
    is_peft_available = False
    PeftModel = None
    LoraLayer = None

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 0. Async Loop Bridge
# ---------------------------------------------------------------------------

class AsyncLoopBridge:
    """
    Spins up a dedicated background thread with a permanent asyncio loop.
    This allows synchronous code (HF Trainer) to call async code (RolloutEngine/vLLM)
    without destroying the event loop and breaking persistent HTTP connection pools.
    """
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return

        def _target() -> None:
            # Create a new loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            # Run forever so httpx connections stay alive across batches
            loop.run_forever()

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()
        # Block main thread until the background loop is actually running
        self._ready.wait()

    def stop(self) -> None:
        if self._loop is not None:
            # Schedule the stop in the loop's thread
            self._loop.call_soon_threadsafe(self._loop.stop)
            # Wait for the thread to finish
            if self._thread is not None:
                self._thread.join()
            self._loop = None
            self._thread = None

    def run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Submit a coroutine to the background loop and block the 
        main thread until the result is ready.
        """
        if self._loop is None:
            raise RuntimeError("AsyncLoopBridge not started. Call start() first.")
        
        # Thread-safe submission to the background loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Block here (synchronously) until the async work is done
        return future.result()


# ---------------------------------------------------------------------------
# 0.5 Generation Gate
# ---------------------------------------------------------------------------

class GenerationGate:
    """
    Forces the Dataset to wait until the Trainer has finished a step (and synced weights)
    before generating the next batch. This ensures strict On-Policy training.
    """
    def __init__(self):
        # Start true so the very first batch can be fetched immediately
        self._can_generate = threading.Event()
        self._can_generate.set()

    def wait_for_signal(self):
        """Called by Dataset before fetching new data."""
        self._can_generate.wait()
        # Once we pass the gate, we clear it. 
        # It won't open again until the Callback sets it.
        self._can_generate.clear()

    def signal_step_complete(self):
        """Called by Callback after weight sync."""
        self._can_generate.set()


# ---------------------------------------------------------------------------
# 1. Dataset Adapter
# ---------------------------------------------------------------------------

class LudicIterableDataset(IterableDataset):
    """
    Bridges the async BatchSource to synchronous HF Trainer.
    
    Yields WHOLE EPISODES (List[SAWItem]), not individual steps.
    This ensures that when HF Trainer requests 'batch_size=8', it gets 
    8 full episodes (containing ~80 steps total), not 8 individual steps.
    
    Features:
    1. Uses `AsyncLoopBridge` to prevent connection pool deadlocks.
    2. Uses `GenerationGate` to ensure data generation waits for weight syncs (On-Policy).
    3. Groups raw steps by Rollout ID.
    """
    def __init__(self, batch_source: BatchSource, bridge: AsyncLoopBridge, gate: GenerationGate):
        self.batch_source = batch_source
        self.bridge = bridge
        self.gate = gate

    def __iter__(self):
        while True:
            # --- BLOCKING WAIT ---
            # Wait for the previous training step & sync to finish.
            self.gate.wait_for_signal()
            
            try:
                # Fetch new batch (using the NEW synced weights)
                saw_batch = self.bridge.run_sync(self.batch_source.next_batch())
                
                # --- Group items by Rollout ID ---
                # We need to reconstruct episodes so the Collator can treat 
                # "One Episode" as "One Example".
                episodes: Dict[str, List[SAWItem]] = defaultdict(list)
                for item in saw_batch.items:
                    r_id = item.meta.get("rollout_id", "unknown")
                    episodes[r_id].append(item)
                
                # Yield lists of items (each list is one full episode)
                for r_id, items in episodes.items():
                    # Ensure steps are sorted by index within the episode
                    items.sort(key=lambda x: x.meta.get("step_index", 0))
                    yield items

            except Exception as e:
                logger.error(f"Error fetching batch from source: {e}")
                raise e


# ---------------------------------------------------------------------------
# 2. Collator
# ---------------------------------------------------------------------------

class LudicDataCollator:
    """
    Collates a list of Episodes (List[List[SAWItem]]) into a single flat batch tensor.
    
    Since the Dataset now yields Lists (Episodes), the input to __call__ 
    is a List of Lists. We flatten this into a single large batch of steps.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, episodes: List[List[SAWItem]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            episodes: A list of 'batch_size' episodes. 
                      Each episode is a list of SAWItems.
        """
        if not episodes:
            raise ValueError("Cannot collate empty list of episodes")

        # 1. Flatten the list of lists into a single list of items
        #    e.g., batch_size=8 rollouts -> ~80 total steps
        all_items: List[SAWItem] = [item for episode in episodes for item in episode]

        if not all_items:
             raise ValueError("Episodes contained no steps")

        # 2. Determine batch shape from the flattened list
        lengths = [len(it.input_ids) for it in all_items]
        max_len = max(lengths)
        total_steps = len(all_items)
        
        # 3. Allocate tensors (CPU)
        # Note: dim 0 is now Total Steps across all episodes
        input_ids = torch.full((total_steps, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((total_steps, max_len), dtype=torch.long)
        action_mask = torch.zeros((total_steps, max_len), dtype=torch.float32)
        weights = torch.zeros((total_steps,), dtype=torch.float32)

        # 4. Fill
        for i, it in enumerate(all_items):
            L = len(it.input_ids)
            input_ids[i, :L] = torch.tensor(it.input_ids, dtype=torch.long)
            attention_mask[i, :L] = torch.tensor(it.attention_mask, dtype=torch.long)
            action_mask[i, :L] = torch.tensor(it.action_mask, dtype=torch.float32)
            weights[i] = float(it.weight)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
            "weight": weights,
        }


# ---------------------------------------------------------------------------
# 3. Sync Callback (LoRA Aware)
# ---------------------------------------------------------------------------

class VLLMSyncCallback(TrainerCallback):
    """
    Pushes weights to vLLM at the end of N steps and signals the GenerationGate.
    If using LoRA, merges adapters on-the-fly before pushing.
    """
    def __init__(self, client: ChatClient, model: nn.Module, gate: GenerationGate, sync_every_steps: int = 1):
        self.client = client
        self.model = model
        self.gate = gate
        self.sync_every_steps = sync_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        # 1. Check if we need to sync
        if state.global_step % self.sync_every_steps == 0:
            self._push_weights()
        
        # 2. IMPORTANT: Signal the dataset that it is safe to generate the NEXT batch.
        # We do this after the push_weights() blocks and returns.
        self.gate.signal_step_complete()

    def _get_merged_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        params = {}
        
        unwrapped = model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # Standard Model
        if not (is_peft_available and isinstance(unwrapped, PeftModel)):
            for name, p in unwrapped.named_parameters():
                if p.requires_grad:
                    params[name] = p.detach()
            return params

        # LoRA (PeftModel)
        for name, module in unwrapped.named_modules():
            if isinstance(module, LoraLayer):
                if module.active_adapter not in module.lora_A:
                    continue
                
                adapter_name = module.active_adapter
                
                # Find base weight
                if hasattr(module, "get_base_layer"):
                    base_layer = module.get_base_layer()
                elif hasattr(module, "weight"):
                    base_layer = module
                else:
                    continue 

                # Calculate merged weight: W_base + (B @ A) * scaling
                w_base = base_layer.weight.detach()
                lora_A = module.lora_A[adapter_name].weight.detach()
                lora_B = module.lora_B[adapter_name].weight.detach()
                scaling = module.scaling[adapter_name]

                delta = (lora_B @ lora_A) * scaling
                merged_weight = w_base + delta
                
                # Clean name for vLLM (remove base_model.model prefix)
                clean_name = name.replace("base_model.model.", "model.")
                if not clean_name.startswith("model."):
                     clean_name = f"model.{clean_name}"
                
                params[f"{clean_name}.weight"] = merged_weight

        return params

    def _push_weights(self):
        if not hasattr(self.client, "push_update_atomic"):
            return
        try:
            params = self._get_merged_params(self.model)
            if params:
                self.client.push_update_atomic(params)
        except Exception as e:
            logger.warning(f"Failed to sync weights to vLLM: {e}")


# ---------------------------------------------------------------------------
# 4. Ludic HF Trainer
# ---------------------------------------------------------------------------

class LudicHFTrainer(Trainer):
    """
    HF Trainer for Ludic RL. 
    - Data from BatchSource via AsyncLoopBridge
    - Loss via RLAlgorithm
    - Optional Reference Model support
    """
    def __init__(
        self,
        model: nn.Module,
        rl_algorithm: RLAlgorithm,
        batch_source: BatchSource,
        client: ChatClient,
        args: TrainingArguments,
        pad_token_id: int,
        ref_model: Optional[nn.Module] = None,
        **kwargs
    ):
        if args.dataloader_num_workers > 0:
            raise ValueError("LudicHFTrainer requires dataloader_num_workers=0")
        
        args.remove_unused_columns = False # Preserve custom columns

        self.rl_algorithm = rl_algorithm
        self.ref_model = ref_model

        # Move ref model to device
        if self.ref_model:
            self.ref_model.eval()
            self.ref_model.to(args.device)

        # --- SETUP ASYNC INFRASTRUCTURE ---
        # 1. Start the persistent background loop
        self.bridge = AsyncLoopBridge()
        self.bridge.start()
        
        # 2. Initialize the Gate to control On-Policy flow
        self.gate = GenerationGate()

        # 3. Pass bridge and gate to Dataset
        train_dataset = LudicIterableDataset(
            batch_source, 
            bridge=self.bridge, 
            gate=self.gate
        )
        # ----------------------------------

        data_collator = LudicDataCollator(pad_token_id=pad_token_id)
        
        sync_callback = VLLMSyncCallback(
            client=client, 
            model=model, 
            gate=self.gate, # Callback releases the gate after sync
            sync_every_steps=getattr(args, "sync_every_steps", 1)
        )
        callbacks = kwargs.get("callbacks", []) + [sync_callback]

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            **kwargs
        )

    # --- CLEANUP HOOK ---
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """
        Wrap the training loop to ensure the background thread is killed
        even if training crashes or finishes early.
        """
        try:
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        finally:
            if hasattr(self, 'bridge'):
                self.bridge.stop()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Run Reference Model (if required)
        if getattr(self.rl_algorithm, "requires_ref_model", False):
            with torch.no_grad():
                ref_context = nullcontext()
                ref_model_to_use = self.ref_model

                # Optimization: If no ref_model provided but model is LoRA, use base weights
                if ref_model_to_use is None and is_peft_available and isinstance(model, PeftModel):
                    ref_model_to_use = model
                    ref_context = model.disable_adapter()

                if ref_model_to_use is None:
                     raise ValueError("RLAlgorithm requires ref_model, but none provided.")

                with ref_context:
                    ref_outputs = ref_model_to_use(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"]
                    )
                    inputs["ref_logits"] = ref_outputs.logits.detach()

        # 2. Compute RL Loss
        loss, stats = self.rl_algorithm.compute_loss(model, inputs)

        # 3. Log
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({f"rl/{k}": v for k, v in stats.items()})

        return (loss, None) if return_outputs else loss
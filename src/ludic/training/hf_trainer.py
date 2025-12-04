from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional
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


# ---------------------------------------------------------------------------
# 1. Dataset Adapter
# ---------------------------------------------------------------------------

class LudicIterableDataset(IterableDataset):
    """
    Bridges the async BatchSource to synchronous HF Trainer.
    Blocks the main thread to fetch batches via asyncio.run().
    """
    def __init__(self, batch_source: BatchSource):
        self.batch_source = batch_source

    def __iter__(self):
        while True:
            try:
                # Fetch one "macro-batch" from the source
                saw_batch = asyncio.run(self.batch_source.next_batch())
            except Exception as e:
                logger.error(f"Error fetching batch from source: {e}")
                raise e
            
            # Yield individual items to HF DataCollator
            yield from saw_batch.items


# ---------------------------------------------------------------------------
# 2. Collator (Inlined Logic)
# ---------------------------------------------------------------------------

class LudicDataCollator:
    """
    Collates a list of SAWItems into a dictionary of tensors.
    Pads input_ids to the max length in the batch.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: List[SAWItem]) -> Dict[str, torch.Tensor]:
        if not items:
            raise ValueError("Cannot collate empty list of SAWItems")

        # 1. Determine batch shape
        lengths = [len(it.input_ids) for it in items]
        max_len = max(lengths)
        batch_size = len(items)
        
        # 2. Allocate tensors (CPU)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        action_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
        weights = torch.zeros((batch_size,), dtype=torch.float32)

        # 3. Fill
        for i, it in enumerate(items):
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
    Pushes weights to vLLM at the end of N steps.
    If using LoRA, merges adapters on-the-fly before pushing.
    """
    def __init__(self, client: ChatClient, model: nn.Module, sync_every_steps: int = 1):
        self.client = client
        self.model = model
        self.sync_every_steps = sync_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.sync_every_steps == 0:
            self._push_weights()

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
    - Data from BatchSource
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

        train_dataset = LudicIterableDataset(batch_source)
        data_collator = LudicDataCollator(pad_token_id=pad_token_id)
        
        sync_callback = VLLMSyncCallback(
            client=client, 
            model=model, 
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
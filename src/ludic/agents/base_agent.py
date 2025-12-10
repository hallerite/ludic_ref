from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Mapping

import torch

from ludic.types import SamplingArgs, Observation, Info, Message
from ludic.inference.client import ChatClient
from ludic.inference.sampling import SamplingConfig, resolve_sampling_args
from ludic.context.base import ContextStrategy
from ludic.parsers import Parser, ParseResult

class Agent:
    """
    A stateful, logical actor that bundles inference, context, and parsing.
    
    It holds a reference to a (potentially shared) ChatClient and manages
    its own internal state via its ContextStrategy.
    """
    name: str = "agent"

    def __init__(
        self, 
        *, 
        client: ChatClient, 
        model: str,
        ctx: ContextStrategy,
        parser: Parser
    ) -> None:
        """
        Initializes the Agent.
        
        Args:
            client: The ChatClient for inference.
            model: The model name this agent should use.
            ctx: An instance of a ContextStrategy for managing memory.
            parser: An instance of a Parser for decoding actions.
        """
        self._client = client
        self._model = model
        self._ctx = ctx
        self._parser = parser
        self.last_info: Dict[str, Any] = {}

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Resets the agent's internal context."""
        self._ctx.reset(system_prompt=system_prompt)
        
    def on_env_reset(self, obs: Observation, info: Info):
        """Called by the protocol *after* env.reset()."""
        self._ctx.on_env_reset(obs, info)
        
    def on_after_step(self, obs: Observation, info: Info):
        """Called by the protocol *after* env.step()."""
        self._ctx.on_after_step(obs, info)

    async def act(
        self,
        sampling_args: SamplingArgs,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ParseResult, str, Dict[str, Any]]:
        """
        Runs the think -> act -> parse cycle based on current context.
        
        This method does *not* take obs/info, as those are fed to the
        agent via on_env_reset() and on_after_step().
        
        Args:
            sampling_args: The sampling configuration for this step.
            timeout_s: Optional timeout for the inference call.
            
        Returns:
            A tuple of (ParseResult, raw_action_text, client_info_dict).
        """
        # 1. Think (prepare prompt messages from context)
        messages: List[Message] = self._ctx.on_before_act()
        
        # 2. Act (run inference)
        sampling: SamplingConfig = resolve_sampling_args(sampling_args)
        coro = self._client.complete(
            model=self._model,
            messages=messages,
            sampling=sampling,
        )
        if timeout_s is None:
            resp, client_info = await coro
        else:
            resp, client_info = await asyncio.wait_for(coro, timeout=timeout_s)

        self.last_info = dict(client_info)
        
        # Also merge token IDs from the response if they exist
        if resp.prompt_token_ids is not None:
            self.last_info["prompt_token_ids"] = resp.prompt_token_ids
        if resp.completion_token_ids is not None:
            self.last_info["completion_token_ids"] = resp.completion_token_ids
        
        # 3. Update memory with the agent's own response
        self._ctx.on_after_act(resp)
        
        # 4. Parse (format the raw text action)
        raw_action = resp.text
        parse_result = self._parser(raw_action)
        
        return parse_result, raw_action, self.last_info

    def push_policy_update(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[str] = None,
    ) -> str:
        """Pushes updated policy parameters to the underlying runtime."""
        if not hasattr(self._client, "sync_weights"):
            raise RuntimeError(
                "Underlying ChatClient does not support policy weight updates "
                "(missing sync_weights)."
            )
        return self._client.sync_weights(
            params,
            timeout_s=timeout_s,
            version=version,
        )
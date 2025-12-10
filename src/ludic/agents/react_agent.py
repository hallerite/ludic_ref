from __future__ import annotations
import json
import inspect
import logging
from typing import Callable, List, Tuple, Dict, Any

from ludic.agents.base_agent import Agent
from ludic.types import SamplingArgs
from ludic.parsers import ParseResult

logger = logging.getLogger(__name__)

class ReActAgent(Agent):
    """
    An agent that implements the ReAct pattern: 
    [Think -> Tool Call] * n -> Act.

    It supports an execution loop where the model can call auxiliary tools 
    multiple times before emitting a final answer for the environment.

    If the max_react_steps limit is reached, it forces a final generation 
    attempt without tools to produce a valid environment action.
    """
    def __init__(self, tools: List[Callable], max_react_steps: int = 5, **kwargs):
        """
        Args:
            tools: List of python functions the agent can call.
            max_react_steps: Maximum number of internal think/tool loops.
            **kwargs: Passed to base Agent.
        """
        super().__init__(**kwargs)
        self.max_react_steps = max_react_steps
        
        # Map function names to the actual callables
        self.tool_map = {t.__name__: t for t in tools}
        
        # Auto-generate OpenAI schemas (simple version)
        self.tool_schemas = [self._func_to_schema(t) for t in tools]
        
        # Safety check: Context must explicitly flag support for tools
        if not self._ctx.supports_tools:
            raise TypeError(f"ReActAgent requires a context with supports_tools=True. ")

    async def act(
        self, 
        sampling_args: SamplingArgs, 
        timeout_s: float | None = None
    ) -> Tuple[ParseResult, str, Dict[str, Any]]:
        
        # 1. Setup Sampling Args
        sargs = sampling_args.copy()
        extras = sargs.get("extras", {}).copy()
        extras["tools"] = self.tool_schemas
        # Optional: force tool usage or let model decide ("auto")
        extras["tool_choice"] = "auto" 

        # --- FORCE PROMPT TOKEN RETURN ---
        # For the sake of simplicity, we need the inference engine to return the 
        # formatted prompt token ids. Dealing with chat templates and tool injection
        # manually is brittle and heavy. By forcing the server to return the IDs it 
        # used, we get ground-truth training data without having to pass the tokenizer instance.
        # TODO: In the future, we want to support inference engines that 
        #       do not support this.
        if "extra_body" not in extras:
            extras["extra_body"] = {}
        extras["extra_body"]["return_token_ids"] = True

        sargs["extras"] = extras

        last_info = {}
        
        # 2. ReAct Loop
        for step_i in range(self.max_react_steps):
            
            # --- THE SHOT CLOCK LOGIC ---
            # If we are on the last step, force a final answer.
            is_final_try = (step_i == self.max_react_steps - 1)
            
            messages = self._ctx.on_before_act()

            if is_final_try:
                # A. Strip tools so it CANNOT call them
                if "tools" in sargs["extras"]:
                    del sargs["extras"]["tools"]
                    if "tool_choice" in sargs["extras"]:
                        del sargs["extras"]["tool_choice"]
                
                # B. Force the agent to wrap up
                # We inject a temporary "system" instruction into the prompt
                # (Note: We don't save this to self._ctx, just for this one call)
                messages = messages + [{
                    "role": "user", 
                    "content": (
                        "You have exhausted your reasoning steps. "
                        "You must output your final move now."
                    )
                }]

            # 3. Inference
            # We call client directly to bypass base Agent structure which assumes single-turn
            from ludic.inference.sampling import resolve_sampling_args
            resolved_sampling = resolve_sampling_args(sargs)
            
            resp, info = await self._client.complete(
                model=self._model,
                messages=messages,
                sampling=resolved_sampling
            )
            
            # Manually merge token IDs from the typed Response object into the info dict.
            # This ensures they persist through the pipeline to the Rollout/Step.
            last_info = dict(info)
            if resp.prompt_token_ids is not None:
                last_info["prompt_token_ids"] = resp.prompt_token_ids
            if resp.completion_token_ids is not None:
                last_info["completion_token_ids"] = resp.completion_token_ids

            # Extract content
            # vLLM/OpenAI specific structure
            raw_choice = info["raw_response"]["choices"][0]
            message_data = raw_choice["message"]
            content = message_data.get("content")
            tool_calls = message_data.get("tool_calls")

            # 4. Handle Final Panic Move
            if is_final_try:
                # We forced it to output text. Return whatever it gave us.
                final_text = content or ""
                parse_result = self._parser(final_text)
                
                # Update context so the agent remembers its final decision
                # We pass None for tool_calls because tools were disabled
                self._ctx.add_assistant_step(final_text, None)
                
                return parse_result, final_text, last_info

            # 5. Normal Logic (Update Context & Check Tools)
            self._ctx.add_assistant_step(content, tool_calls)

            if tool_calls:
                # --- EXECUTION PHASE ---
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    args_json = tc["function"]["arguments"]
                    call_id = tc["id"]
                    
                    result_str = ""
                    if fn_name in self.tool_map:
                        try:
                            # Parse JSON args
                            fn_args = json.loads(args_json)
                            # Call Python function
                            obs = self.tool_map[fn_name](**fn_args)
                            result_str = str(obs)
                        except Exception as e:
                            result_str = f"Error executing tool {fn_name}: {e}"
                            logger.warning(result_str)
                    else:
                        result_str = f"Error: Tool {fn_name} not found."

                    # Record Observation
                    self._ctx.add_tool_result(call_id, fn_name, result_str)
                
                # Continue loop -> Model sees result and thinks again
                continue
            
            else:
                # --- FINAL ANSWER PHASE ---
                # No tool calls means the model is talking to the user/env.
                # We expect the content to contain the action here.
                # The format is defined by self._parser (XML, JSON, Regex, etc).
                final_text = content or ""
                parse_result = self._parser(final_text)
                return parse_result, final_text, last_info

        # This should technically be unreachable due to the is_final_try block,
        # but good for safety.
        fallback_msg = "Error: Loop logic failure."
        return self._parser(fallback_msg), fallback_msg, last_info

    def _func_to_schema(self, f: Callable) -> Dict[str, Any]:
        """
        Minimal schema generator. 
        For production, use Pydantic to inspect signature types accurately.
        """
        sig = inspect.signature(f)
        params = {}
        required_params = []
        
        for name, param in sig.parameters.items():
            # A very naive type mapping
            p_type = "string"
            if param.annotation == int: p_type = "integer"
            elif param.annotation == float: p_type = "number"
            elif param.annotation == bool: p_type = "boolean"
            
            params[name] = {"type": p_type}
            
            if param.default == inspect.Parameter.empty:
                required_params.append(name)

        return {
            "type": "function",
            "function": {
                "name": f.__name__,
                "description": f.__doc__ or "No description provided.",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required_params
                }
            }
        }
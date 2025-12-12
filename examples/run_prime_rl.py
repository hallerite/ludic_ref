# run_experiment.py
import asyncio
import logging

from ludic.training.prime_bridge import LudicConfig, PrimeOrchestrator
from prime_rl.utils.pydantic_config import parse_argv

from ludic.training.rollout_engine import RolloutEngine, GRPOBatchSource
from ludic.training.credit_assignment import GroupNormalizedReturn
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest

from examples.envs.tic_tac_toe import TicTacToeEnv
from ludic.inference.vllm_client import VLLMChatClient
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.parsers import xml_move_parser


# -------------------------------------------------------------------------
# 1. Environment registry
# -------------------------------------------------------------------------

env_registry = {
    "tictactoe_agent_starts": lambda **kw: TicTacToeEnv(agent_starts=True, **kw),
    "tictactoe_opp_starts": lambda **kw: TicTacToeEnv(agent_starts=False, **kw),
}


# -------------------------------------------------------------------------
# 2. Protocol registry
# -------------------------------------------------------------------------

def create_single_agent_protocol(**kwargs):
    """
    Factory for the SingleAgentSyncProtocol used by Ludic's RolloutEngine.

    NOTE:
    - enable_weight_updates=False because Prime's admin_clients handle weight sync.
    - The prompt below is just a placeholder; you likely want to combine
      `TicTacToeEnv().suggested_sysprompt` with stricter XML instructions.
    """
    client = VLLMChatClient(
        host="127.0.0.1",
        port=8000,
        enable_weight_updates=False,
    )

    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-7B-Instruct",
        ctx=FullDialog(),
        parser=xml_move_parser,
    )

    prompt = TicTacToeEnv().suggested_sysprompt or ""
    # You can optionally append extra instructions for XML format here.

    return SingleAgentSyncProtocol(agent=agent, prompt=prompt)


protocol_registry = {"single_agent_xml": create_single_agent_protocol}


# -------------------------------------------------------------------------
# 3. Main orchestration
# -------------------------------------------------------------------------

async def main():
    # 3.1 Parse combined Prime+Ludic config (from TOML / CLI)
    #
    # Expected TOML (rough sketch):
    # [orchestrator]
    # env_kind = "tictactoe_agent_starts"
    # protocol_kind = "single_agent_xml"
    #
    # [orchestrator.sampling]
    # temperature = 0.7
    # max_tokens = 64
    #
    # and standard Prime RL fields (output_dir, num_train_workers, rollouts_per_example, ...)
    config = parse_argv(LudicConfig)

    # 3.2 Build the Ludic rollout engine
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=str(config.output_dir / "logs" / "rollouts.jsonl"),
    )

    # 3.3 Build credit assigner (GRPO-style group-normalized return)
    credit_assigner = GroupNormalizedReturn(normalize_adv=True)

    # 3.4 Define GRPO base requests: one per "group" / prompt
    def base_requests_fn():
        # Number of groups: each group will be expanded to G rollouts by GRPOBatchSource,
        # where G = config.rollouts_per_example (must divide batch_size).
        num_groups = config.batch_size // config.rollouts_per_example

        # Use some reproducible base seed if available; otherwise 0
        base_seed = getattr(config, "seed", 0)

        return [
            RolloutRequest(
                env=EnvSpec(kind=config.env_kind, kwargs=config.env_kwargs),
                protocol=ProtocolSpec(
                    kind=config.protocol_kind,
                    kwargs=config.protocol_kwargs,
                ),
                num_episodes=1,
                sampling_args={
                    "temperature": config.sampling.temperature,
                    "max_tokens": config.sampling.max_tokens,
                    "seed": base_seed + i,
                    # Crucial: vLLM must return token IDs for Ludic → SAW → Prime
                    "extras": {
                        "extra_body": {"return_token_ids": True}
                    },
                },
            )
            for i in range(num_groups)
        ]

    # 3.5 Choose a BatchSource (GRPO here; RolloutBatchSource would also work)
    batch_source = GRPOBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=base_requests_fn,
        group_size=config.rollouts_per_example,          # the "G" in GRPO
        max_steps=config.seq_len,
        concurrency=min(128, config.batch_size),
        timeout_s=None,
        retokenize=False,
        tokenize=None,
    )

    # 3.6 Build and run the PrimeOrchestrator bridge
    #
    # PrimeOrchestrator:
    #   - pulls SAWBatches from batch_source
    #   - syncs weights with Prime trainer via broadcast dir
    #   - converts to Prime MicroBatches and writes rank_i.pt files
    orchestrator = PrimeOrchestrator(config=config, batch_source=batch_source)
    await orchestrator.loop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())

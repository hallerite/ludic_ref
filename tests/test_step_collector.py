from ludic.interaction.step_collector import TraceCollector
from ludic.types import Step

def test_trace_collector_separation():
    """Ensure collector separates steps by agent_id into distinct rollouts."""
    collector = TraceCollector(env_name="test_env")

    # Agent A steps
    step_a1 = Step(index=0, prev_obs="a1", action="act_a1", next_obs="o", reward=1.0, truncated=False, terminated=False)
    step_a2 = Step(index=1, prev_obs="a2", action="act_a2", next_obs="o", reward=1.0, truncated=False, terminated=True)
    
    # Agent B steps
    step_b1 = Step(index=0, prev_obs="b1", action="act_b1", next_obs="o", reward=-1.0, truncated=False, terminated=True)

    collector.add("agent_A", step_a1)
    collector.add("agent_B", step_b1) # Interleaved
    collector.add("agent_A", step_a2)

    rollouts = collector.extract_rollouts()

    assert len(rollouts) == 2
    
    # Sort by agent_id to verify deterministically
    rollouts.sort(key=lambda r: r.meta["agent_id"])
    
    r_a = rollouts[0]
    assert r_a.meta["agent_id"] == "agent_A"
    assert r_a.meta["env_name"] == "test_env"
    assert len(r_a.steps) == 2
    assert r_a.steps[0] == step_a1
    assert r_a.steps[1] == step_a2

    r_b = rollouts[1]
    assert r_b.meta["agent_id"] == "agent_B"
    assert len(r_b.steps) == 1
    assert r_b.steps[0] == step_b1

def test_trace_collector_ignores_empty_agents():
    """Ensure agents registered but not active produce no rollouts."""
    collector = TraceCollector()
    # No steps added
    assert len(collector.extract_rollouts()) == 0
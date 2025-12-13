import pytest

from ludic.envs.dataset_qa_env import DatasetQAEnv


def test_dataset_env_runs_one_step_and_grades_correctly():
    sample = {"question": "What is 1 + 1?", "answer": "2"}
    env = DatasetQAEnv(sample)

    obs_info = env.reset(seed=0)
    obs, info = obs_info["agent_0"]

    assert "1 + 1" in obs
    assert info["question_id"] == "0"

    outcome = env.step({"agent_0": "2"})["agent_0"]

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.reward == pytest.approx(1.0)
    assert outcome.info["correct"] is True
    assert outcome.info["parsed_answer"] == "2"
    assert outcome.info["target_answer"] == "2"


def test_custom_parser_handles_gsm8k_and_math_formats():
    sample = {"question": "Box this", "answer": "\\boxed{7}"}
    def gsm_target_parser(text: str) -> str:
        cleaned = text.strip()
        if "####" in cleaned:
            cleaned = cleaned.split("####")[-1].strip()
        if "\\boxed{" in cleaned:
            cleaned = cleaned.replace("\\boxed{", "").replace("}", "")
        return cleaned.strip()

    env = DatasetQAEnv(sample, target_parser=gsm_target_parser)
    env.reset(seed=0)

    outcome = env.step({"agent_0": "7"})["agent_0"]

    assert outcome.info["parsed_answer"] == "7"
    assert outcome.info["target_answer"] == "7"
    assert outcome.info["correct"] is True


def test_sequential_sampling_without_shuffle():
    sample = {"question": "Q1", "answer": "a"}
    env = DatasetQAEnv(sample)

    obs1, _ = env.reset(seed=0)["agent_0"]
    obs2, _ = env.reset()["agent_0"]
    assert obs1 == "Q1"
    assert obs2 == "Q1"

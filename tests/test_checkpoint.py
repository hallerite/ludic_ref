from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ludic.training.checkpoint import CheckpointConfig, CheckpointManager


class DummyHFModel(nn.Module):
    """
    Minimal HF-like model exposing save_pretrained for checkpoint tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.saved_state_dict: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not used
        return self.layer(x)

    def save_pretrained(self, save_directory: str | Path, *, state_dict=None) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.saved_state_dict = state_dict
        torch.save(state_dict or self.state_dict(), save_path / "pytorch_model.bin")
        (save_path / "config.json").write_text(json.dumps({"model_class": "DummyHFModel"}))


def test_checkpoint_manager_saves_hf_layout(tmp_path: Path) -> None:
    model = DummyHFModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    cfg = CheckpointConfig(
        output_dir=str(tmp_path / "ckpts"),
        every_n_steps=1,
        max_to_keep=2,
        save_optimizer=True,
    )
    manager = CheckpointManager(cfg)

    ckpt_path = manager.maybe_save(
        model,
        optimizer=optimizer,
        step=1,
        metadata={"algorithm": "unit-test"},
    )

    assert ckpt_path is not None
    assert (ckpt_path / "pytorch_model.bin").exists()
    assert (ckpt_path / "config.json").exists()
    assert (ckpt_path / "optimizer.pt").exists()

    with (ckpt_path / "trainer_state.json").open("r") as fh:
        state = json.load(fh)
    assert state["step"] == 1
    assert state["algorithm"] == "unit-test"


def test_checkpoint_manager_prunes_old_checkpoints(tmp_path: Path) -> None:
    model = DummyHFModel()
    cfg = CheckpointConfig(
        output_dir=str(tmp_path / "ckpts"),
        every_n_steps=1,
        max_to_keep=1,
        save_optimizer=False,
    )
    manager = CheckpointManager(cfg)

    for step in [1, 2, 3]:
        manager.save(model, optimizer=None, step=step, metadata=None)

    saved_dirs = sorted((tmp_path / "ckpts").glob("step_*"))
    assert len(saved_dirs) == 1
    assert saved_dirs[0].name == "step_000003"

    with (tmp_path / "ckpts" / "latest.json").open("r") as fh:
        latest = json.load(fh)
    assert latest["step"] == 3


def test_checkpoint_manager_loads_model_and_optimizer(tmp_path: Path) -> None:
    model = DummyHFModel()
    for p in model.parameters():
        p.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Seed optimizer state so we can verify restore.
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state[p]["test_buf"] = torch.tensor([42.0])

    cfg = CheckpointConfig(output_dir=str(tmp_path / "ckpts"), every_n_steps=1, save_optimizer=True)
    manager = CheckpointManager(cfg)
    manager.save(model, optimizer=optimizer, step=7, metadata={"tag": "golden"})

    # Fresh model/optimizer with different weights/state
    model2 = DummyHFModel()
    for p in model2.parameters():
        p.data.zero_()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

    meta = manager.load(model2, optimizer=optimizer2, step=7)

    for p in model2.parameters():
        assert torch.allclose(p, torch.ones_like(p))

    restored_states = list(optimizer2.state.values())
    assert restored_states and "test_buf" in restored_states[0]
    assert torch.equal(restored_states[0]["test_buf"], torch.tensor([42.0]))
    assert meta["step"] == 7


def test_checkpoint_manager_loads_hf_sharded_bin_checkpoint(tmp_path: Path) -> None:
    model = DummyHFModel()
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    assert len(keys) >= 2

    ckpt_dir = tmp_path / "sharded_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shard_1 = {keys[0]: state_dict[keys[0]]}
    shard_2 = {keys[1]: state_dict[keys[1]]}

    shard_1_name = "pytorch_model-00001-of-00002.bin"
    shard_2_name = "pytorch_model-00002-of-00002.bin"
    torch.save(shard_1, ckpt_dir / shard_1_name)
    torch.save(shard_2, ckpt_dir / shard_2_name)

    index = {
        "metadata": {},
        "weight_map": {
            keys[0]: shard_1_name,
            keys[1]: shard_2_name,
        },
    }
    (ckpt_dir / "pytorch_model.bin.index.json").write_text(json.dumps(index))
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"step": 123}))

    manager = CheckpointManager(CheckpointConfig(output_dir=str(tmp_path / "unused"), every_n_steps=0))

    model2 = DummyHFModel()
    for p in model2.parameters():
        p.data.zero_()

    meta = manager.load(model2, path=str(ckpt_dir))
    assert meta["step"] == 123

    loaded = model2.state_dict()
    assert torch.equal(loaded[keys[0]], state_dict[keys[0]])
    assert torch.equal(loaded[keys[1]], state_dict[keys[1]])

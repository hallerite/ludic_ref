from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Observation, Info, StepOutcome


_COORD_RE = re.compile(r"^[ABCabc][123]$")


def _coord_to_index(coord: str) -> int:
    """
    Map board coordinates like 'A1', 'b3' → integer index 0..8.

    A/B/C = rows (top/mid/bottom)
    1/2/3 = columns (left/mid/right)

    A1 A2 A3
    B1 B2 B3
    C1 C2 C3
    """
    coord = coord.strip().upper()
    if not _COORD_RE.match(coord):
        raise ValueError(f"Invalid move coordinate: {coord!r}. Expected e.g. 'A1', 'B3'.")

    row_char, col_char = coord[0], coord[1]
    row = {"A": 0, "B": 1, "C": 2}[row_char]
    col = {"1": 0, "2": 1, "3": 2}[col_char]
    return row * 3 + col


@dataclass
class _GameResult:
    winner: Optional[str]  # "X" / "O" / None
    draw: bool


class TicTacToeEnv(SingleAgentEnv):
    """
    Minimal Tic-Tac-Toe environment for Ludic.

    - Agent always plays 'X'
    - Opponent always plays 'O'
    - Agent moves first
    - Opponent plays uniformly random legal moves
    - Actions are coordinates like "A1", "b2", ...
    """

    WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    def __init__(
        self,
        *,
        agent_starts: bool = True,
        show_opponent_move: bool = True,
    ) -> None:
        """
        Args:
            agent_starts:
                If True, agent is always X and moves first.
                (For now we fix marks as X vs O regardless; this flag only
                controls whether the opponent makes an opening random move.)
            show_opponent_move:
                If True, the observation string will mention the last
                opponent move explicitly.
        """
        # SingleAgentEnv defaults the agent identifier to "agent_0"
        super().__init__()
        self.agent_mark = "X"
        self.opponent_mark = "O"
        self.agent_starts = agent_starts
        self.show_opponent_move = show_opponent_move

        self._board: List[Optional[str]] = [None] * 9
        self._done: bool = False
        self._last_opponent_move: Optional[str] = None
        self._obs_cache: Observation = ""

    # ------------------------------------------------------------------
    # Env interface (SingleAgentEnv implementation)
    # ------------------------------------------------------------------

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return (
            "You are playing Tic-Tac-Toe as 'X'.\n"
            "The board has rows A (top), B (middle), C (bottom) and "
            "columns 1 (left), 2 (middle), 3 (right).\n"
            "A move is written like A1, B2, C3, etc."
        )

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        if seed is not None:
            random.seed(seed)

        self._board = [None] * 9
        self._done = False
        self._last_opponent_move = None

        if not self.agent_starts:
            self._opponent_random_move()

        obs = self._render_obs()
        info: Info = {
            "agent_mark": self.agent_mark,
            "opponent_mark": self.opponent_mark,
        }
        return obs, info

    def env_step(self, action: str) -> StepOutcome:
        """
        `action` is a coordinate string like "A1", "b3", etc.

        Reward:
          +1.0  agent wins
          +0.5  draw
           0.0  agent loses
          -1.0  illegal move
           0.0  non-terminal step
        """
        if self._done:
            raise RuntimeError(
                "TicTacToeEnv.step() called after episode is done. Call reset()."
            )

        info: Info = {}

        # 1) Parse & apply agent move
        try:
            idx = _coord_to_index(action)
        except ValueError as e:
            self._done = True
            obs = self._render_obs()
            info.update(
                {
                    "illegal_move": True,
                    "error": str(e),
                    "agent_move": action,
                }
            )
            return StepOutcome(
                obs=obs,
                reward=-1.0,
                truncated=True,
                terminated=True,
                info=info,
            )

        if self._board[idx] is not None:
            self._done = True
            obs = self._render_obs()
            coord_norm = action.strip().upper()
            info.update(
                {
                    "illegal_move": True,
                    "error": f"Cell {coord_norm} is already occupied.",
                    "agent_move": coord_norm,
                }
            )
            return StepOutcome(
                obs=obs,
                reward=-1.0,
                truncated=True,
                terminated=True,
                info=info,
            )

        self._board[idx] = self.agent_mark
        info["agent_move"] = action.strip().upper()

        # 2) Check terminal after agent move
        result = self._check_game_over()
        if result is not None:
            self._done = True
            reward = self._result_to_reward(result)
            obs = self._render_obs()
            info["result"] = self._result_label(result)
            return StepOutcome(
                obs=obs,
                reward=reward,
                truncated=False,
                terminated=True,
                info=info,
            )

        # 3) Opponent move
        opp_idx = self._opponent_random_move()
        if opp_idx is not None:
            row = "ABC"[opp_idx // 3]
            col = "123"[opp_idx % 3]
            opp_coord = f"{row}{col}"
            self._last_opponent_move = opp_coord
            info["opponent_move"] = opp_coord

        # 4) Check terminal after opponent move
        result = self._check_game_over()
        if result is not None:
            self._done = True
            reward = self._result_to_reward(result)
            obs = self._render_obs()
            info["result"] = self._result_label(result)
            return StepOutcome(
                obs=obs,
                reward=reward,
                truncated=False,
                terminated=True,
                info=info,
            )

        # Non-terminal
        obs = self._render_obs()
        return StepOutcome(
            obs=obs,
            reward=0.0,
            truncated=False,
            terminated=False,
            info=info,
        )

    def env_current_obs(self) -> Observation:
        return self._obs_cache

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _empty_cells(self) -> List[int]:
        return [i for i, v in enumerate(self._board) if v is None]

    def _check_game_over(self) -> Optional[_GameResult]:
        # Winner?
        for a, b, c in self.WIN_LINES:
            m = self._board[a]
            if m is not None and m == self._board[b] == self._board[c]:
                return _GameResult(winner=m, draw=False)

        # Draw?
        if all(cell is not None for cell in self._board):
            return _GameResult(winner=None, draw=True)

        return None

    def _result_to_reward(self, result: _GameResult) -> float:
        if result.draw:
            return 0.5
        if result.winner == self.agent_mark:
            return 1.0
        # Loss (opponent wins)
        return 0.0

    def _result_label(self, result: _GameResult) -> str:
        if result.draw:
            return "draw"
        if result.winner == self.agent_mark:
            return "win"
        if result.winner == self.opponent_mark:
            return "loss"
        return "unknown"

    def _opponent_random_move(self) -> Optional[int]:
        empty = self._empty_cells()
        if not empty:
            return None
        idx = random.choice(empty)
        self._board[idx] = self.opponent_mark
        return idx

    def _render_obs(self) -> Observation:
        """Render board + available moves as a text observation."""
        cells: List[str] = []
        for i in range(9):
            mark = self._board[i]
            if mark is not None:
                cells.append(f" {mark} ")
            else:
                row = "ABC"[i // 3]
                col = "123"[i % 3]
                cells.append(f"[{row}{col}]")

        sep = "\n" + ("----+----+----") + "\n"
        board_lines = (
            f"{cells[0]}|{cells[1]}|{cells[2]}"
            f"{sep}"
            f"{cells[3]}|{cells[4]}|{cells[5]}"
            f"{sep}"
            f"{cells[6]}|{cells[7]}|{cells[8]}"
        )

        empty_indices = self._empty_cells()
        available_moves = []
        for i in empty_indices:
            row = "ABC"[i // 3]
            col = "123"[i % 3]
            available_moves.append(f"{row}{col}")
        available_moves_str = ", ".join(sorted(available_moves))

        parts = []
        if self.show_opponent_move and self._last_opponent_move is not None:
            parts.append(
                f"Opponent ({self.opponent_mark}) played at {self._last_opponent_move}."
            )

        parts.append("Current Tic-Tac-Toe board:")
        parts.append(board_lines)
        parts.append(f"Available moves: {available_moves_str}")
        parts.append(
            f"You are '{self.agent_mark}'. Choose your next move (e.g. A1, B2, C3)."
        )

        obs = "\n".join(parts)
        self._obs_cache = obs
        return obs

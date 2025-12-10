from __future__ import annotations
from typing import Callable, List, Optional

from ludic.training.types import (
    BatchSource, SAWBatch, RolloutRequest, 
    CreditAssigner, TokenizeFn
)
from .rollout_engine import RolloutEngine

class RolloutBatchSource(BatchSource):
    """
    Synchronous BatchSource. 
    It holds the Engine and blocks the Trainer while generating data.
    """
    def __init__(
        self,
        *,
        orchestrator: RolloutEngine,
        credit_assigner: CreditAssigner,
        requests_fn: Callable[[], List[RolloutRequest]],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        retokenize: bool = False,
        tokenize: Optional[TokenizeFn] = None,
    ) -> None:
        self._engine = orchestrator
        self._credit_assigner = credit_assigner
        self._requests_fn = requests_fn
        self._max_steps = max_steps
        self._timeout_s = timeout_s
        self._concurrency = concurrency
        self._retokenize = retokenize
        self._tokenize = tokenize

    async def next_batch(self) -> SAWBatch:
        """
        Pull requests -> Generate (blocking) -> Return Batch.
        """
        requests = self._requests_fn()
        return await self._engine.generate_batch(
            requests=requests,
            max_steps=self._max_steps,
            credit_assigner=self._credit_assigner,
            timeout_s=self._timeout_s,
            concurrency=self._concurrency,
            retokenize=self._retokenize,
            tokenize=self._tokenize,
        )
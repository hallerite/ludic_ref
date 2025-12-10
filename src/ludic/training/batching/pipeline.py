from __future__ import annotations

import pickle
import logging
import asyncio
import redis
from typing import List, Callable, Optional
from dataclasses import replace

from ludic.training.types import (
    BatchSource, 
    SAWBatch, 
    SAWItem, 
    RolloutRequest, 
    CreditAssigner,
    TokenizeFn
)
from ludic.inference.client import VersionedClient
from .rollout_engine import RolloutEngine

logger = logging.getLogger(__name__)

class PipelineBatchSource(BatchSource):
    """
    Trainer-side component.
    Pulls completed, pre-processed SAWItems from a Redis queue.
    
    This decouples the Trainer from the generation latency. The Trainer
    simply blocks on the queue until data arrives.
    """
    def __init__(
        self, 
        redis_url: str, 
        queue_key: str = "ludic_queue", 
        batch_size: int = 4,
        poll_timeout: int = 1
    ):
        self.r = redis.from_url(redis_url)
        self.queue_key = queue_key
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout

    async def next_batch(self) -> SAWBatch:
        """
        Blocking fetch from Redis. Returns a SAWBatch once enough items are pulled.
        """
        items: List[SAWItem] = []
        
        while len(items) < self.batch_size:
            # BLPOP blocks until data is available, preventing busy-loops.
            # We use a short timeout to allow the loop to check for exit signals/cancellation.
            raw_data = self.r.blpop(self.queue_key, timeout=self.poll_timeout)
            
            if raw_data:
                # raw_data is tuple (queue_name, payload_bytes)
                payload = raw_data[1]
                try:
                    # The Actor has already done the tokenization and credit assignment.
                    # We just deserialize the final training sample.
                    saw_item: SAWItem = pickle.loads(payload)
                    items.append(saw_item)
                except Exception as e:
                    logger.error(f"Failed to deserialize SAWItem from Redis: {e}")
            else:
                # Timeout occurred, loop again or yield to event loop
                await asyncio.sleep(0.01)
                continue

        # Calculate basic batch stats for logging
        avg_reward = 0.0
        if items:
            total_r = sum(it.meta.get("total_reward", 0.0) for it in items)
            avg_reward = total_r / len(items)

        meta = {
            "batch_size": len(items),
            "avg_total_reward": avg_reward,
            "source": "pipeline_redis"
        }

        return SAWBatch(items=items, meta=meta)


# -------------------------------------------------------------------------
# The Actor Loop
# -------------------------------------------------------------------------

async def run_pipeline_actor(
    engine: RolloutEngine,
    requests_fn: Callable[[], List[RolloutRequest]],
    credit_assigner: CreditAssigner,
    redis_url: str,
    queue_key: str = "ludic_queue",
    max_steps: int = 10,
    concurrency: int = 4,
    retokenize: bool = False,
    tokenize: Optional[TokenizeFn] = None,
    client: Optional[VersionedClient] = None,
):
    """
    Actor-side component. Runs an infinite loop to:
    1. Fetch intent via requests_fn.
    2. Poll the runtime (via client) for the current policy version.
    3. Tag requests with that version.
    4. Delegate generation AND collation to the shared Engine.
    5. Push the resulting SAWItems to Redis.
    """
    r_conn = redis.from_url(redis_url)
    logger.info(f"Pipeline Actor connected to Redis at {redis_url}")
    
    while True:
        # 1. Get Intent (and apply Strategy via requests_fn)
        requests = requests_fn()
        if not requests:
            await asyncio.sleep(1.0)
            continue
        
        # 2. Fetch Version (Clock Sync)
        current_ver = 0
        if client:
             current_ver = await client.get_policy_version()

        # 3. Tag Requests
        # We explicitly inject policy_version so the Trainer can filter later.
        tagged_requests = []
        for req in requests:
            new_meta = req.meta.copy()
            new_meta["policy_version"] = current_ver
            
            # Use replace to safely copy the dataclass with updated meta
            tagged_requests.append(replace(req, meta=new_meta))

        # 4. Execute Generation & Collation
        # We call generate_batch instead of generate_rollouts.
        # This ensures that tokenization, masking, and credit assignment logic
        # are mathematically identical to the synchronous RolloutBatchSource.
        try:
            saw_batch = await engine.generate_batch(
                requests=tagged_requests,
                max_steps=max_steps,
                credit_assigner=credit_assigner,
                concurrency=concurrency,
                retokenize=retokenize,
                tokenize=tokenize
            )
        except Exception as e:
            logger.error(f"Error in actor generation loop: {e}")
            await asyncio.sleep(1.0)
            continue
        
        if not saw_batch.items:
            await asyncio.sleep(0.1)
            continue

        # 5. Serialize & Push
        # We unbundle the batch so the Trainer can re-bundle them into 
        # whatever batch size it prefers.
        pipe = r_conn.pipeline()
        count = 0

        for item in saw_batch.items:
            try:
                # Pickle the SAWItem (input_ids, masks, weight, meta)
                pipe.rpush(queue_key, pickle.dumps(item))
                count += 1
            except Exception as e:
                logger.error(f"Failed to serialize item: {e}")

        # Bulk push to Redis for efficiency
        if count > 0:
            try:
                pipe.execute()
                logger.debug(f"Pushed {count} items (v{current_ver}) to {queue_key}")
            except redis.RedisError as e:
                logger.error(f"Redis pipeline error: {e}")
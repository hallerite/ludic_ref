from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional

from ludic.training.types import SAWBatch, SAWItem


ReducerKind = Literal["mean", "sum", "count_true"]
NormalizeBy = Literal["samples", "rollouts", None]


@dataclass
class Reducer:
    """
    Declarative aggregator over item metadata.

    - kind: aggregation strategy ("mean" | "sum" | "count_true")
    - source: dotted path into SAWItem.meta (e.g., "parse_error", "result")
              or a callable that accepts SAWItem and returns a value.
    - transform: optional post-processing of the raw value (e.g., lambda v: v == "win").
    - normalize_by: None | "samples" | "rollouts" — override denominator to
      normalize by total samples or total rollouts.
    """

    kind: ReducerKind
    source: str | Callable[[SAWItem], object]
    transform: Optional[Callable[[object], object]] = None
    normalize_by: NormalizeBy = None


def _get_value_from_source(item: SAWItem, source: str | Callable[[SAWItem], object]) -> object:
    if callable(source):
        return source(item)
    # Treat strings as dotted paths into item.meta
    current: object = item.meta
    for part in source.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _apply_transform(value: object, transform: Optional[Callable[[object], object]]) -> object:
    return transform(value) if transform else value


def aggregate_stats(
    micro_stats_list: List[Dict[str, float]],
    saw_batches: List[SAWBatch],
    *,
    reducers: Mapping[str, Reducer] | None = None,
) -> Dict[str, float]:
    """
    Aggregate micro-batch stats with batch-level metadata and optional reducers.

    By default, micro-batch stats are averaged across micro-batches, except
    keys prefixed with ``gpu_`` which are reduced using ``max`` (since they
    typically represent peaks).

    Core outputs (always present):
        - num_samples: total SAWItems across batches
        - num_rollouts: total agent trajectories across batches
        - avg_total_reward: item-weighted average of batch.meta["avg_total_reward"]
        - avg_completion_length: item-weighted average of batch.meta["avg_completion_length"]
    Plus any keys produced by `reducers`.
    """
    if not micro_stats_list:
        return {}

    # 1) Standard loss/grad stats
    all_keys = {k for ms in micro_stats_list for k in ms.keys()}
    max_keys = {k for k in all_keys if k.startswith("gpu_")}

    agg_stats: Dict[str, float] = {}
    sum_counts: Dict[str, int] = {}
    for k in all_keys:
        if k in max_keys:
            agg_stats[k] = float("-inf")
        else:
            agg_stats[k] = 0.0
            sum_counts[k] = 0

    for micro_stats in micro_stats_list:
        for k, v in micro_stats.items():
            if k in max_keys:
                agg_stats[k] = max(agg_stats[k], v)
            else:
                agg_stats[k] += v
                sum_counts[k] += 1

    for k in list(agg_stats.keys()):
        if k in max_keys:
            if agg_stats[k] == float("-inf"):
                del agg_stats[k]
            continue
        denom = sum_counts.get(k, 0)
        agg_stats[k] = agg_stats[k] / float(denom) if denom > 0 else 0.0

    # 2) Batch metadata stats
    total_samples = 0.0
    total_rollouts = 0.0
    total_reward_sum = 0.0
    total_completion_len = 0.0

    for batch in saw_batches:
        num_items = float(len(batch.items))
        total_samples += num_items

        # Each batch has N rollouts (agent trajectories)
        batch_rollouts = float(
            batch.meta.get("num_rollouts")
            or batch.meta.get("batch_size", 0.0)  # fallback for older producers
        )
        total_rollouts += batch_rollouts

        # Weighted reward average
        avg_reward = float(batch.meta.get("avg_total_reward", 0.0))
        total_reward_sum += avg_reward * num_items

        # Weighted completion length average
        avg_completion_len = float(batch.meta.get("avg_completion_length", 0.0))
        total_completion_len += avg_completion_len * num_items

    agg_stats["num_samples"] = total_samples
    agg_stats["num_rollouts"] = total_rollouts

    if total_samples > 0:
        agg_stats["avg_total_reward"] = total_reward_sum / total_samples
        agg_stats["avg_completion_length"] = total_completion_len / total_samples
    else:
        agg_stats["avg_total_reward"] = 0.0
        agg_stats["avg_completion_length"] = 0.0

    # 3) Custom reducers (optional)
    if reducers:
        # Flatten items once for convenience
        items: List[SAWItem] = [item for batch in saw_batches for item in batch.items]

        for name, reducer in reducers.items():
            values: List[object] = []
            for item in items:
                raw = _get_value_from_source(item, reducer.source)
                if raw is None:
                    continue
                values.append(_apply_transform(raw, reducer.transform))

            result: float = 0.0
            if reducer.kind == "mean":
                if values:
                    result = float(sum(float(v) for v in values) / len(values))
            elif reducer.kind == "sum":
                result = float(sum(float(v) for v in values))
            elif reducer.kind == "count_true":
                result = float(sum(1.0 for v in values if bool(v)))
            else:
                raise ValueError(f"Unknown reducer kind: {reducer.kind}")

            if reducer.normalize_by == "samples":
                denom = total_samples
            elif reducer.normalize_by == "rollouts":
                denom = total_rollouts
            else:
                denom = None

            if denom:
                result = result / denom if denom > 0 else 0.0

            agg_stats[name] = result

    return agg_stats


def apply_reducers_to_records(
    records: Iterable[Mapping[str, Any]],
    reducers: Mapping[str, Reducer],
    *,
    sample_count: Optional[float] = None,
    rollout_count: Optional[float] = None,
) -> Dict[str, float]:
    """
    Apply reducers to a list of dict-like records (e.g., eval results).

    - records: iterable of dicts with metric fields.
    - reducers: mapping name -> Reducer.
    - sample_count / rollout_count: optional explicit denominators for
      normalize_by. Defaults to len(list(records)).
    """
    items = list(records)
    totals = {
        "samples": float(sample_count) if sample_count is not None else float(len(items)),
        "rollouts": float(rollout_count) if rollout_count is not None else float(len(items)),
    }

    out: Dict[str, float] = {}
    for name, reducer in reducers.items():
        values: List[object] = []
        for rec in items:
            raw = rec
            for part in reducer.source.split(".") if isinstance(reducer.source, str) else []:
                if not isinstance(raw, dict) or part not in raw:
                    raw = None
                    break
                raw = raw[part]
            if callable(reducer.source):
                raw = reducer.source(rec)  # type: ignore[arg-type]
            if raw is None:
                continue
            values.append(_apply_transform(raw, reducer.transform))

        result: float = 0.0
        if reducer.kind == "mean":
            if values:
                result = float(sum(float(v) for v in values) / len(values))
        elif reducer.kind == "sum":
            result = float(sum(float(v) for v in values))
        elif reducer.kind == "count_true":
            result = float(sum(1.0 for v in values if bool(v)))
        else:
            raise ValueError(f"Unknown reducer kind: {reducer.kind}")

        if reducer.normalize_by == "samples":
            denom = totals["samples"]
        elif reducer.normalize_by == "rollouts":
            denom = totals["rollouts"]
        else:
            denom = None

        if denom:
            result = result / denom if denom > 0 else 0.0

        out[name] = result

    return out

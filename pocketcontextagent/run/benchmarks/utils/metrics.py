"""Provider-reported episode measurements; unavailable or incomplete totals are null."""

import math
from collections import Counter


def _number(value: object) -> int | float | None:
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _tokens(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _sum(values: list) -> int | float | None:
    return sum(values) if values and all(value is not None for value in values) else None


def _max(values: list) -> int | float | None:
    return max(values) if values and all(value is not None for value in values) else None


def _average(values: list) -> float | None:
    total = _sum(values)
    return total / len(values) if total is not None else None


def _usage(response: dict) -> dict:
    usage = response.get("usage") or {}
    input_tokens = _tokens(usage.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _tokens(usage.get("input_tokens"))
    output_tokens = _tokens(usage.get("completion_tokens"))
    if output_tokens is None:
        output_tokens = _tokens(usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": _tokens(usage.get("total_tokens")),
        "cached_input_tokens": _tokens(details.get("cached_tokens")),
    }


def episode_metrics(
    messages: list[dict], *, model_stats: dict | None = None, latency_seconds: float | None = None
) -> dict:
    """Count unique model responses, including compaction and persisted format errors.

    Response IDs deduplicate copies. Without IDs only repeated object references are
    deduplicated: two identical responses can be separate billable calls. Missing
    usage in any call makes that token total/peak unknown, not a measured zero.
    Cached tokens are a subset of input tokens, never added to the total.
    Cost comes only from model_stats; latency must be measured by the caller.
    """
    responses, calls = {}, {}
    for message in messages:
        extra = message.get("extra") or {}
        response = message if message.get("object") in ("response", "chat.completion") else extra.get("response")
        key = ("message", id(message))
        if isinstance(response, dict) or message.get("role") == "assistant" or "response" in extra:
            if isinstance(response, dict):
                key = ("response", response["id"]) if response.get("id") else ("object", id(response))
            measured = responses.setdefault(key, {})
            for name, value in _usage(response if isinstance(response, dict) else {}).items():
                if value is not None:
                    measured.setdefault(name, value)
        for index, action in enumerate(extra.get("actions") or []):
            parts = action.get("command", "").split(maxsplit=1)
            name = action.get("type") or (parts[0] if parts else "unknown")
            calls.setdefault(action.get("tool_call_id") or (key, index), name)
    expected_calls = _tokens((model_stats or {}).get("api_calls")) or 0
    for index in range(max(0, expected_calls - len(responses))):
        responses[("missing", index)] = {}
    for usage in responses.values():
        if "tokens" not in usage:
            total = _sum([usage.get("input_tokens"), usage.get("output_tokens")])
            if total is not None:
                usage["tokens"] = total
    metrics = {
        "model_calls": len(responses),
        "calls_with_usage": sum(bool(usage) for usage in responses.values()),
        "tool_call_counts": dict(Counter(calls.values())),
        "total_tool_calls": len(calls),
        "cost_usd": _number((model_stats or {}).get("instance_cost"))
        if (model_stats or {}).get("cost_available", True)
        else None,
        "latency_seconds": _number(latency_seconds),
    }
    metrics["calls_without_usage"] = metrics["model_calls"] - metrics["calls_with_usage"]
    for name in ("input_tokens", "output_tokens", "tokens", "cached_input_tokens"):
        values = [usage.get(name) for usage in responses.values()]
        metrics[f"total_{name}"] = _sum(values)
        metrics[f"calls_with_{'total_tokens' if name == 'tokens' else name}"] = sum(
            value is not None for value in values
        )
        if name in ("input_tokens", "tokens"):
            metrics["peak_input_tokens" if name == "input_tokens" else "peak_total_tokens"] = _max(values)
    return metrics


def _task_tokens(metric: dict, key: str) -> int | None:
    # Legacy trajectories reported token zeros even when no usage was available.
    if metric.get("calls_with_usage") == 0:
        return None
    if key == "total_tokens" and key not in metric:
        return _sum([_tokens(metric.get("total_input_tokens")), _tokens(metric.get("total_output_tokens"))])
    return _tokens(metric.get(key))


def summarize_results(results: list[dict]) -> dict:
    """Summarize attempted tasks without equating submission with correctness.

    Only result['correct'] booleans are grades. The primary success rate requires
    every task to be graded; graded_success_rate describes the graded subset.
    Totals, peaks, and averages require complete measurements; explicit coverage
    counts identify missing data. No input results means null measurements.
    Cost uses model_stats.instance_cost, not provider usage prices.
    """
    metrics = [result.get("metrics") or {} for result in results]
    grades = [result["correct"] for result in results if type(result.get("correct")) is bool]
    counts = Counter()
    for metric in metrics:
        counts.update(metric.get("tool_call_counts") or {})
    costs = [
        _number((result.get("model_stats") or {}).get("instance_cost"))
        if (result.get("model_stats") or {}).get("cost_available", True)
        else None
        for result in results
    ]
    latencies = [_number(metric.get("latency_seconds")) for metric in metrics]
    peaks = [_task_tokens(metric, "peak_input_tokens") for metric in metrics]
    tool_calls = [metric.get("total_tool_calls", 0) for metric in metrics]
    summary = {
        "tasks": len(results),
        "completed": sum(result.get("status") == "completed" for result in results),
        "graded_tasks": len(grades),
        "correct_tasks": sum(grades),
        "incorrect_tasks": len(grades) - sum(grades),
        "ungraded_tasks": len(results) - len(grades),
        "success_rate": sum(grades) / len(results) if results and len(grades) == len(results) else None,
        "graded_success_rate": sum(grades) / len(grades) if grades else None,
        "peak_input_tokens": _max(peaks),
        "avg_peak_input_tokens": _average(peaks),
        "tasks_with_peak_input_tokens": sum(value is not None for value in peaks),
        "avg_tool_calls_per_task": _average(tool_calls) if metrics else 0,
        "tool_call_counts": dict(counts),
        "avg_research_tool_calls_per_task": _average(
            [
                total - (metric.get("tool_call_counts") or {}).get("submit", 0)
                for total, metric in zip(tool_calls, metrics)
            ]
        )
        if metrics
        else 0,
        "cost_usd": _sum(costs),
        "avg_cost_usd_per_task": _average(costs),
        "tasks_with_cost": sum(value is not None for value in costs),
        "total_latency_seconds": _sum(latencies),
        "avg_latency_seconds_per_task": _average(latencies),
        "tasks_with_latency": sum(value is not None for value in latencies),
    }
    for name in ("input_tokens", "output_tokens", "tokens", "cached_input_tokens"):
        values = [_task_tokens(metric, f"total_{name}") for metric in metrics]
        summary[f"total_{name}"] = _sum(values)
        label = "total_tokens" if name == "tokens" else name
        summary[f"avg_{label}_per_task"] = _average(values)
        summary[f"tasks_with_{label}"] = sum(value is not None for value in values)
    return summary

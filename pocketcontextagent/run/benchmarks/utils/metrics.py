"""Provider-reported token usage and semantic tool counts across an agent episode."""

import json
from collections import Counter


def episode_metrics(messages: list[dict]) -> dict:
    responses, calls = {}, {}
    for message in messages:
        extra = message.get("extra", {})
        response = extra.get("response")
        if isinstance(response, dict):
            responses.setdefault(response.get("id") or json.dumps(response, sort_keys=True), response)
        for index, action in enumerate(extra.get("actions", [])):
            parts = action.get("command", "").split(maxsplit=1)
            name = action.get("type") or (parts[0] if parts else "unknown")
            calls.setdefault(action.get("tool_call_id") or f"{id(message)}:{index}", name)
    usage = [response["usage"] for response in responses.values() if response.get("usage")]
    return {
        "peak_input_tokens": max((item.get("prompt_tokens", item.get("input_tokens", 0)) for item in usage), default=0),
        "peak_total_tokens": max((item.get("total_tokens", 0) for item in usage), default=0),
        "total_input_tokens": sum(item.get("prompt_tokens", item.get("input_tokens", 0)) for item in usage),
        "total_output_tokens": sum(item.get("completion_tokens", item.get("output_tokens", 0)) for item in usage),
        "calls_with_usage": len(usage),
        "tool_call_counts": dict(Counter(calls.values())),
        "total_tool_calls": len(calls),
    }


def summarize_results(results: list[dict]) -> dict:
    metrics = [result["metrics"] for result in results]
    counts = Counter()
    for metric in metrics:
        counts.update(metric["tool_call_counts"])
    return {
        "tasks": len(results),
        "completed": sum(result["status"] == "completed" for result in results),
        "peak_input_tokens": max((metric["peak_input_tokens"] for metric in metrics), default=0),
        "avg_peak_input_tokens": sum(metric["peak_input_tokens"] for metric in metrics) / len(metrics)
        if metrics
        else 0,
        "avg_tool_calls_per_task": sum(metric["total_tool_calls"] for metric in metrics) / len(metrics)
        if metrics
        else 0,
        "tool_call_counts": dict(counts),
        "avg_research_tool_calls_per_task": (
            sum(metric["total_tool_calls"] - metric["tool_call_counts"].get("submit", 0) for metric in metrics)
            / len(metrics)
            if metrics
            else 0
        ),
        "cost_usd": sum(result["model_stats"]["instance_cost"] for result in results),
    }

"""Append-only ReAct, using the no-memory BrowseComp-Plus prompt from ACM.

Source: lixiaochuan2020/agentic-context-management@f06f90e728af8580a4515812425c1620144145a2,
MIT licensed. Original ReAct paradigm: arXiv:2210.03629. See comparisons/upstream/.
"""

from pocketcontextagent.comparisons.methods.base import ComparisonAgent

SYSTEM_PROMPT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the tools multiple times.

Your context window is {context_window} tokens. Plan your searches accordingly — once context is full the conversation ends immediately. You must provide your answer in the required format before reaching the context limit.
IMPORTANT: Search snippets are often incomplete. Before answering, always retrieve and examine at least one full document using get_document.

Your response should be in the following format:
Explanation: {{{{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}}}
Exact Answer: {{{{your succinct, final answer}}}}
Confidence: {{{{your confidence score between 0% and 100% for your answer}}}}"""


def build_system_prompt(context_window: int) -> str:
    return SYSTEM_PROMPT.format(context_window=context_window)


class ReActAgent(ComparisonAgent):
    """Reasoning/action/observation history without compression or recall tools."""

    method_name = "react"

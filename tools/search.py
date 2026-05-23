"""
tools/search.py
---------------
Tavily-powered web search tool for the Mini AI Agent.

The agent calls `search_web(query)` when it needs real-time information
that is not in the LLM's training data (current events, prices, etc.).

Requires:
  TAVILY_API_KEY set in the environment (via .env).
"""

import os
from typing import Optional


def search_web(query: str) -> str:
    """
    Search the web using the Tavily API and return a concise summary.

    Parameters
    ----------
    query : str
        A natural-language search query, e.g. ``"latest Python 3.13 release notes"``.

    Returns
    -------
    str
        A formatted string containing the search answer and/or a list of
        result snippets, or a descriptive error message on failure.

    Notes
    -----
    - Uses ``search_depth="advanced"`` for richer, more reliable results.
    - Caps results at 3 items to keep the observation short for the LLM.
    - All exceptions are caught and returned as readable error strings so the
      agent loop can continue gracefully (it will report the error as an
      observation and decide what to do next).
    """
    query = query.strip()
    if not query:
        return "Error: Empty search query provided."

    # ── Validate API key before importing the client ──────────────────────────
    api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Error: TAVILY_API_KEY is not set. "
            "Add it to your .env file to enable web search."
        )

    try:
        from tavily import TavilyClient  # lazy import — not needed if key missing

        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_answer=True,       # Tavily synthesises a short answer
            include_raw_content=False, # keep payload small
        )

        parts: list[str] = []

        # 1. If Tavily returned a synthesised answer, use it first.
        answer: Optional[str] = response.get("answer")
        if answer:
            parts.append(f"Summary: {answer}")

        # 2. Append individual result snippets for extra context.
        results: list[dict] = response.get("results", [])
        if results:
            parts.append("Sources:")
            for i, result in enumerate(results, start=1):
                title   = result.get("title", "No title")
                url     = result.get("url", "")
                content = result.get("content", "").strip()
                # Truncate long snippets so the context window stays manageable.
                snippet = content[:300] + "…" if len(content) > 300 else content
                parts.append(f"  [{i}] {title}\n      {url}\n      {snippet}")

        if not parts:
            return "Search returned no results."

        return "\n".join(parts)

    except ImportError:
        return (
            "Error: 'tavily-python' package is not installed. "
            "Run: pip install tavily-python"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error during web search: {type(exc).__name__}: {exc}"


# ── Quick smoke-test when run directly ───────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_query = "What is the latest version of Python?"
    print(f"Query: {test_query}\n")
    print(search_web(test_query))

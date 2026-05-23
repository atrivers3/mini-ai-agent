"""
main.py
-------
Entry point for the Mini AI Agent.

Usage:
  1.  & ".\.venv\Scripts\Activate.ps1"
  2.  pip install -r requirements.txt
  3.  python main.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env before importing the agent,
# so that GEMINI_API_KEY is available when the model is configured.
load_dotenv()

from core.agent import run_agent  # noqa: E402  (import after dotenv)


def main() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║           Mini AI Agent — ReAct Demo            ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Test questions ────────────────────────────────────────────────────────
    # Each entry exercises the ReAct loop: the agent must call the calculator
    # tool at least once before it can answer.
    test_prompts = [
        "What is (123 * 456) + 789?",
        "If a pizza costs $12.50 and I buy 7, how much do I spend in total?",
        "What is 2 to the power of 16?",
    ]

    for prompt in test_prompts:
        answer = run_agent(prompt, verbose=True)
        # run_agent already prints verbose output; the answer is also returned
        # for any downstream use (e.g., a future web API layer).
        _ = answer   # silence unused-variable linters


if __name__ == "__main__":
    main()

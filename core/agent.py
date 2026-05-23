"""
core/agent.py
-------------
Heart of the Mini AI Agent — implements the ReAct (Reason + Act) loop.

Flow per iteration:
  1.  Send conversation history to Gemini.
  2.  Parse the JSON response into { thought, tool_name, tool_input, is_final_answer }.
  3a. If is_final_answer == False  →  execute the requested tool,
      append the observation, and loop.
  3b. If is_final_answer == True   →  return the final answer string.
"""

import json
import os
import re
import sys
from typing import Any

# import google.generativeai as genai
from google import genai

# ── Tool registry ─────────────────────────────────────────────────────────────
# Import every tool the agent is allowed to use and register it here.
# The key must match exactly what the LLM is told to use in tool_name.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.calculator import calculate  # noqa: E402

TOOL_REGISTRY: dict[str, Any] = {
    "calculator": calculate,
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a meticulous AI assistant that solves problems step-by-step.

STRICT OUTPUT CONTRACT
======================
Every single response you produce MUST be a valid JSON object — nothing else.
No markdown, no code fences, no extra prose. Pure JSON only.

The JSON object must contain exactly these four keys:

{
  "thought":         "<your internal reasoning about what to do next>",
  "tool_name":       "<name of the tool to call, or null if answering directly>",
  "tool_input":      "<string argument to pass to the tool, or null>",
  "is_final_answer": <true | false>
}

RULES
=====
1.  "thought"  — Always fill this in. Explain your reasoning before acting.
2.  "tool_name" / "tool_input" — Required when is_final_answer is false.
    Set both to null when is_final_answer is true.
3.  "is_final_answer" — Set to true ONLY when you have enough information
    to give a complete, correct answer to the user. When true, put the
    full answer inside "thought" (tool_name and tool_input must be null).
4.  Never guess a numeric result — always call the calculator tool.

AVAILABLE TOOLS
===============
- calculator : Evaluates a safe arithmetic expression string.
               Example input: "(3 + 5) * 12 / 2"

EXAMPLE INTERACTION
===================
User asks: "What is 17 * 8?"

Turn 1 — you output:
{
  "thought": "I need to multiply 17 by 8. I'll use the calculator.",
  "tool_name": "calculator",
  "tool_input": "17 * 8",
  "is_final_answer": false
}

System feeds back: Observation: Result: 136

Turn 2 — you output:
{
  "thought": "The calculator returned 136. I can now answer the user.",
  "tool_name": null,
  "tool_input": null,
  "is_final_answer": true
}
"""

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 5  # Safety cap — prevents runaway loops


# ── Helpers ──────────────────────────────────────────────────────────────────
def _build_model() -> genai.Client:
    """Initialise and return the Gemini GenAI Client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example → .env and add your key."
        )
    return genai.Client(api_key=api_key)
    # genai.configure(api_key=api_key)
    # client = genai.Client()
    # response = client.models.generate_content(
    #     model='gemini-2.5-flash',
    #     contents='Your prompt here'
    # )


def _extract_json(raw: str) -> dict:
    """
    Extract the first JSON object from a raw model response string.

    Gemini sometimes wraps JSON inside markdown code fences even when told
    not to, so we strip those first, then fall back to a regex scan.
    """
    # Strip markdown code fences (```json … ``` or ``` … ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Attempt direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex fallback: grab the first { … } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model response:\n{raw}")


def _dispatch_tool(tool_name: str, tool_input: str) -> str:
    """Look up and call a tool from the registry."""
    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        return (
            f"Error: Unknown tool '{tool_name}'. "
            f"Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    try:
        return tool_fn(tool_input)
    except Exception as exc:  # noqa: BLE001
        return f"Error while running tool '{tool_name}': {exc}"


# ── Main agent loop ───────────────────────────────────────────────────────────
def run_agent(user_prompt: str, verbose: bool = True) -> str:
    """
    Run the ReAct loop for a given user prompt.
    """
    client = _build_model()

    if verbose:
        print("\n" + "═" * 60)
        print(f"   USER: {user_prompt}")
        print("═" * 60)

    # 1. Initialize the chat session ONCE outside the loop.
    # The new SDK automatically manages history correctly within the session.
    chat = client.chats.create(
        model='gemini-2.5-flash',
        config={'system_instruction': SYSTEM_PROMPT}
    )

    # We send the initial prompt to kickstart the interaction.
    current_message = user_prompt

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for iteration in range(1, MAX_ITERATIONS + 1):
        if verbose:
            print(f"\n── Iteration {iteration}/{MAX_ITERATIONS} " + "─" * 30)

        # 2. Send the message to the active chat session
        response = chat.send_message(current_message)
        raw_text = response.text.strip()

        # 3. Parse the mandatory JSON response
        try:
            parsed = _extract_json(raw_text)
        except ValueError as exc:
            print(f"  [PARSE ERROR] {exc}")
            return "Agent error: model did not return valid JSON."

        thought         = parsed.get("thought", "")
        tool_name       = parsed.get("tool_name")
        tool_input      = parsed.get("tool_input")
        is_final_answer = parsed.get("is_final_answer", False)

        if verbose:
            print(f"  💭 THOUGHT: {thought}")

        # 3a. Final answer — we're done.
        if is_final_answer:
            if verbose:
                print(f"\n  ✅ FINAL ANSWER: {thought}")
                print("═" * 60 + "\n")
            return thought

        # 3b. Tool call — execute and feed the observation back.
        if tool_name:
            if verbose:
                print(f"  🔧 TOOL CALL: {tool_name}({tool_input!r})")

            observation = _dispatch_tool(tool_name, tool_input or "")

            if verbose:
                print(f"  📋 OBSERVATION: {observation}")

            # Set the observation string as the next message to process in the loop
            current_message = f"Observation: {observation}"
        else:
            # Model returned is_final_answer=False but gave no tool — treat as done.
            if verbose:
                print("  ⚠️  No tool specified and not a final answer. Returning thought.")
            return thought

    # ── Iteration cap reached ─────────────────────────────────────────────────
    fallback = (
        "Agent reached the maximum number of iterations "
        f"({MAX_ITERATIONS}) without a final answer."
    )
    if verbose:
        print(f"\n  ⚠️  {fallback}")
    return fallback

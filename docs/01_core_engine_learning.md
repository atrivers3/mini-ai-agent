# Core Engine Deep-Dive: How the ReAct Loop Works

> **Audience:** You — the developer building this agent.  
> **File being explained:** [`core/agent.py`](../core/agent.py)

---

## 1. What is ReAct?

**ReAct** stands for **Re**ason + **Act**. It is a prompting pattern where a
Large Language Model (LLM) interleaves two kinds of output:

| Phase | What happens |
|-------|-------------|
| **Reason** | The model articulates its *thought* — what it knows, what it still needs, what it plans to do. |
| **Act**    | The model names a *tool* to call and the *input* for that tool. |

The results of actions (called **Observations**) are fed back to the model
as new context, allowing it to reason again before acting again.
This cycle repeats until the model decides it has enough information to give
a **Final Answer**.

```
User Prompt
    │
    ▼
┌───────────────────────────────────────┐
│  LLM: Reason  →  Act (pick tool)      │
└───────────────────────────────────────┘
    │
    ▼ tool result
┌───────────────────────────────────────┐
│  Observation fed back to LLM          │
└───────────────────────────────────────┘
    │
    ▼  (loop…)
┌───────────────────────────────────────┐
│  LLM: Reason  →  Final Answer ✅      │
└───────────────────────────────────────┘
```

---

## 2. The System Prompt — Forcing Structured Thought

The first critical piece is the **system prompt** in `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = """
Every single response you produce MUST be a valid JSON object…

{
  "thought":         "…",
  "tool_name":       "…",
  "tool_input":      "…",
  "is_final_answer": true | false
}
"""
```

### Why JSON?

Normally an LLM outputs free-form prose. The JSON schema **forces the model to
externalise its reasoning** (`"thought"`) into a machine-readable field *before*
it is allowed to act. This achieves two things simultaneously:

1. **Structured reasoning** — the model cannot skip ahead to an action without
   writing its thought first (analogous to showing work in math).
2. **Programmatic parseability** — our Python code can branch on
   `is_final_answer`, read `tool_name`, and extract `tool_input` without
   fragile regex scraping of natural-language output.

---

## 3. The `while` Loop — Step by Step

```python
for iteration in range(1, MAX_ITERATIONS + 1):
```

We use a `for` range (acting as a `while` with a built-in counter) so the
maximum iteration count is guaranteed.

### Step-by-step inside each iteration:

```
Iteration N
│
├─ 1. CALL MODEL
│       chat = model.start_chat(history=history[:-1])
│       response = chat.send_message(last_message)
│
├─ 2. PARSE JSON
│       parsed = _extract_json(response.text)
│       ── extracts: thought, tool_name, tool_input, is_final_answer
│
├─ 3a. is_final_answer == True?
│       └─ return parsed["thought"]   ← DONE
│
└─ 3b. is_final_answer == False?
        ├─ Call tool: observation = _dispatch_tool(tool_name, tool_input)
        ├─ Append model response to history (role="model")
        └─ Append observation to history  (role="user", "Observation: …")
            └─ Loop back to iteration N+1
```

### The conversation `history` list

```python
history = [
  {"role": "user",  "parts": ["What is 123 * 456?"]},         # seed
  {"role": "model", "parts": ['{"thought":…, "tool_name":…}']},  # iter 1
  {"role": "user",  "parts": ["Observation: Result: 56088"]},    # observation
  {"role": "model", "parts": ['{"thought":…, "is_final_answer": true}']}, # iter 2
]
```

Each observation is injected *as a user message* so the model treats it as
ground-truth external feedback, not as its own prior output. This is the key
mechanism: the model can only "see" a tool result once our code has executed
it and reflected the observation back.

---

## 4. JSON Parsing — Why It Can Be Tricky

Despite instructions, Gemini may occasionally wrap the JSON in a markdown code
fence (` ```json … ``` `). The `_extract_json()` helper handles this:

```python
def _extract_json(raw: str) -> dict:
    # 1. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    
    # 2. Try direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Regex fallback: grab first { … } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    …
```

This defensive parsing means the agent is robust even when the model is slightly
non-compliant with its own instructions.

---

## 5. Why We Cap at `MAX_ITERATIONS = 5`

### The risks of an uncapped loop

| Risk | Consequence |
|------|------------|
| Model hallucinates a tool that never returns `is_final_answer: true` | Infinite loop, runaway API costs |
| A tool call fails repeatedly | Agent spins forever trying to recover |
| Prompt drift — context window fills up | Increasingly degraded reasoning quality |

### Why 5 specifically?

For the current tool set (just a calculator), any solvable problem needs
**at most 2–3 iterations** (one or two tool calls + one final answer turn).
`5` gives comfortable headroom for multi-step problems while still catching
runaway loops quickly.

> **Rule of thumb:** `MAX_ITERATIONS` ≈ (expected tool calls) × 2 + 1.
> Increase it as you add more complex tools (e.g., web search chains).

---

## 6. The Tool Registry

```python
TOOL_REGISTRY: dict[str, Any] = {
    "calculator": calculate,
}
```

The registry is a plain Python dictionary mapping the **string name** the LLM
uses in `"tool_name"` to the **actual callable**. To add a new tool:

1. Write the function (e.g., `tools/web_search.py`).
2. Import it in `core/agent.py`.
3. Add it to `TOOL_REGISTRY`.
4. Describe it in the `SYSTEM_PROMPT`'s "AVAILABLE TOOLS" section.

The agent will automatically start using it — no loop logic changes required.

---

## 7. End-to-End Trace Example

**Prompt:** `"What is (123 * 456) + 789?"`

```
Iteration 1
  Model output (JSON):
    { "thought": "I need to calculate (123 * 456) + 789. I'll use the calculator.",
      "tool_name": "calculator",
      "tool_input": "(123 * 456) + 789",
      "is_final_answer": false }

  Tool executed: calculate("(123 * 456) + 789")
  Observation:   "Result: 56877"

Iteration 2
  Model output (JSON):
    { "thought": "The calculator returned 56877. The answer is 56877.",
      "tool_name": null,
      "tool_input": null,
      "is_final_answer": true }

  ✅ Final answer returned: "The calculator returned 56877. The answer is 56877."
```

Two iterations, one tool call, no hallucination. That is the ReAct loop working
exactly as designed.

---

## 8. Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| JSON-only output contract | Eliminates ambiguity; enables reliable parsing |
| `thought` field always required | Forces chain-of-thought; improves accuracy |
| Observations injected as `"user"` role | Treats external data as ground truth |
| `MAX_ITERATIONS` hard cap | Prevents infinite loops and runaway costs |
| AST-based calculator (not `eval`) | Security — prevents code injection via tool input |
| Tool registry dict | Open/Closed: add tools without touching loop logic |

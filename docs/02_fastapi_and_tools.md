# FastAPI Integration & Tool Expansion

> **Audience:** You — the developer building this agent.
> **Files explained:** [`server.py`](../server.py) · [`api/routes.py`](../api/routes.py) · [`api/models.py`](../api/models.py) · [`tools/search.py`](../tools/search.py)

---

## 1. Why FastAPI?

The ReAct loop in `core/agent.py` is a Python function. To make it accessible
to any client (a web UI, a mobile app, another service), we need an **HTTP
interface**. FastAPI is the right choice here for three reasons:

| Reason | Detail |
|--------|--------|
| **Automatic validation** | Pydantic models reject bad requests before your code even runs |
| **Async-first** | Built on Starlette + asyncio — handles many concurrent requests without threads |
| **Auto-docs** | `/docs` (Swagger UI) and `/redoc` are generated for free from your code |

---

## 2. Pydantic Models — Data Integrity at the Border

```
HTTP Request (raw JSON)
       │
       ▼
  Pydantic validates
  ┌───────────────────────────────────────────┐
  │  AgentRequest                             │
  │    prompt:     str  (required, len ≥ 1)  │
  │    session_id: str | None  (optional)     │
  └───────────────────────────────────────────┘
       │  422 if invalid ──────────────────────►  client
       │  ✓ if valid
       ▼
  Your endpoint code runs
       │
       ▼
  ┌───────────────────────────────────────────┐
  │  AgentResponse                            │
  │    answer:  str                           │
  │    status:  "success" | "error"           │
  └───────────────────────────────────────────┘
       │
       ▼
HTTP Response (serialised JSON)
```

### Why does this matter?

Without Pydantic, you'd write manual validation code like:

```python
# ❌ Manual validation — error-prone and verbose
body = await request.json()
if "prompt" not in body or not isinstance(body["prompt"], str):
    return JSONResponse({"error": "prompt required"}, status_code=422)
```

With Pydantic you declare the shape once and get:
- **Type coercion** — `"42"` (string) becomes `42` (int) for `int` fields automatically.
- **Validation errors** with clear field-level messages, automatically returned as
  `422 Unprocessable Entity`.
- **OpenAPI schema** auto-generated from your field definitions and `Field(...)` metadata.

### The `session_id` field

It's `Optional[str] = None` right now, meaning it's accepted but unused.
This is intentional — it's a placeholder for **Phase 3** when you add
per-session conversation memory. Defining the field now means existing clients
can already send it without breaking the API contract later.

---

## 3. The Async Bridge — Why `run_in_executor`?

The ReAct loop in `run_agent()` makes **blocking** HTTP calls (to the Gemini
API and Tavily). FastAPI runs on an `asyncio` event loop — blocking it would
freeze the entire server for every concurrent user.

```python
# ✅ What we do in api/routes.py
answer = await loop.run_in_executor(
    None,            # default ThreadPoolExecutor
    run_agent,       # blocking function
    request.prompt,  # argument
)
```

```
FastAPI event loop (async)
│
├─ Request A arrives  → dispatched to Thread 1 (run_agent running)
├─ Request B arrives  → dispatched to Thread 2 (run_agent running)
├─ Request C arrives  → dispatched to Thread 3 (run_agent running)
│
│   (event loop is FREE to accept more requests while threads work)
│
└─ Thread 1 returns  → Response A sent
   Thread 2 returns  → Response B sent
```

**Alternative:** Rewrite `run_agent` with `async`/`await` and async HTTP
clients (`httpx`). That's more efficient at scale but requires more refactoring.
`run_in_executor` is the pragmatic bridge for now.

---

## 4. CORS Middleware — Why It's Needed

Browsers block cross-origin requests by default (the "same-origin policy").
When your future React/Vue front-end on `http://localhost:3000` calls
`http://localhost:8000/api/ask`, the browser sees a **different origin** and
blocks it unless the server explicitly allows it.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev: allow all origins
    …
)
```

> ⚠️ **Before production:** set `CORS_ORIGINS=https://yourdomain.com` in your
> environment to restrict access.

---

## 5. The New Search Tool — Expanding the Agent's Capabilities

### Without search
The agent was limited to its **training data cutoff**. Ask it about today's news
and it would either hallucinate or refuse.

### With Tavily search
```
User: "Who won the latest Nobel Prize in Physics?"

Iteration 1
  Thought: "I need current information. I'll use the search tool."
  Tool: search("Nobel Prize Physics 2024 winner")
  Observation: "Summary: The 2024 Nobel Prize in Physics was awarded to…"

Iteration 2
  Thought: "I have the answer from the search result."
  is_final_answer: true  →  return answer
```

### Why Tavily over raw Google/DuckDuckGo scraping?

| Feature | Tavily | Raw scraping |
|---------|--------|--------------|
| LLM-optimised snippets | ✅ Yes | ❌ Must parse yourself |
| Synthesised answer | ✅ Yes | ❌ No |
| Rate limits / robots.txt | ✅ Handled | ❌ Your problem |
| Setup complexity | ✅ API key only | ❌ Complex |

### Graceful degradation

`search_web()` **never raises an exception** — all errors are caught and
returned as readable strings. This means the agent loop always gets a valid
observation, even if it says `"Error: TAVILY_API_KEY is not set."`. The agent
can then decide what to do (e.g., tell the user it cannot search right now).

---

## 6. End-to-End Request Flow

```
Client  →  POST /api/ask  {"prompt": "What is the latest LLM news?"}
               │
               ▼
         FastAPI parses & validates via AgentRequest (Pydantic)
               │  422 if bad ──────────────────────────────► Client
               │  ✓ valid
               ▼
         ask_agent() endpoint
               │
               ▼
         asyncio.run_in_executor(run_agent, prompt)
               │                       │
               │              [Thread] ReAct loop
               │                 Iter 1: → Gemini API
               │                 Iter 1: ← JSON {tool:"search",…}
               │                 Iter 1: → Tavily API
               │                 Iter 1: ← observation
               │                 Iter 2: → Gemini API
               │                 Iter 2: ← JSON {is_final_answer:true}
               │                 return answer string
               │
               ▼
         AgentResponse(answer=…, status="success")
               │
               ▼
  Client  ←  200 OK  {"answer": "…", "status": "success"}
```

---

## 7. Running the Server

```powershell
# Install new deps
pip install -r requirements.txt

# Start the dev server (auto-reloads on file changes)
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:**      http://localhost:8000/redoc
- **Health:**     http://localhost:8000/api/health

### Test with curl
```powershell
curl -X POST http://localhost:8000/api/ask `
  -H "Content-Type: application/json" `
  -d '{"prompt": "What is 123 * 456?"}'
```

Expected response:
```json
{"answer": "The result of 123 * 456 is 56088.", "status": "success"}
```

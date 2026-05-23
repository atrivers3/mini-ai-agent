"""
api/routes.py
-------------
FastAPI router — exposes the ReAct agent over HTTP.

Endpoints
---------
POST /ask   Accept a user prompt, run the agent, return the answer.
GET  /health  Lightweight liveness check (useful for container health probes).
"""

import asyncio
from fastapi import APIRouter, HTTPException
from api.models import AgentRequest, AgentResponse
from core.agent import run_agent

router = APIRouter(prefix="/api", tags=["agent"])


# ── POST /api/ask ─────────────────────────────────────────────────────────────
@router.post(
    "/ask",
    response_model=AgentResponse,
    summary="Send a prompt to the AI agent",
    description=(
        "Runs the ReAct (Reason + Act) loop for the given prompt. "
        "The agent may call one or more tools before returning its final answer."
    ),
)
async def ask_agent(request: AgentRequest) -> AgentResponse:
    """
    POST /api/ask

    The core ReAct loop in ``core/agent.py`` is **synchronous** (it uses
    blocking Gemini API calls). We offload it to a thread-pool executor via
    ``asyncio.get_event_loop().run_in_executor`` so the FastAPI event loop
    is never blocked and can continue handling other concurrent requests.

    Parameters
    ----------
    request : AgentRequest
        Validated request body (Pydantic guarantees ``prompt`` is non-empty).

    Returns
    -------
    AgentResponse
        ``{ "answer": "...", "status": "success" | "error" }``
    """
    try:
        loop = asyncio.get_event_loop()

        # run_agent is a blocking function — run it in the default thread pool
        # so it does not stall the async event loop.
        answer: str = await loop.run_in_executor(
            None,           # use the default ThreadPoolExecutor
            run_agent,      # the callable
            request.prompt, # positional arg 1
        )

        # Detect agent-level errors (the loop returns error strings, not exceptions).
        status = "error" if answer.lower().startswith("agent error") else "success"
        return AgentResponse(answer=answer, status=status)

    except EnvironmentError as exc:
        # Missing API key — surface as a clear 503 so the client knows it's
        # a configuration issue, not a bug in the request.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:  # noqa: BLE001
        # Unexpected failures — don't leak stack traces to the client.
        raise HTTPException(
            status_code=500,
            detail=f"Internal agent error: {type(exc).__name__}: {exc}",
        ) from exc


# ── GET /api/health ───────────────────────────────────────────────────────────
@router.get(
    "/health",
    summary="Health check",
    description="Returns 200 OK if the server is running.",
    tags=["meta"],
)
async def health_check() -> dict:
    """Lightweight liveness probe — no external calls."""
    return {"status": "ok", "service": "mini-ai-agent"}

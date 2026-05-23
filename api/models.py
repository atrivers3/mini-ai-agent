"""
api/models.py
-------------
Pydantic models that define the contract between the HTTP client and the API.

Pydantic v2 automatically:
  - Validates incoming JSON against the field types.
  - Rejects requests with missing required fields (returns 422 Unprocessable Entity).
  - Serialises Python objects back to JSON for the response.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """
    Request body for the POST /ask endpoint.

    Fields
    ------
    prompt : str
        The question or task the user wants the agent to solve.
        Must be a non-empty string.
    session_id : str | None
        Optional client-supplied session identifier.
        Can be used in the future for per-session conversation memory.
        Defaults to None if not provided.
    """

    prompt: str = Field(
        ...,                          # required — no default
        min_length=1,
        description="The user's question or task for the agent.",
        examples=["What is 42 * 17?", "Who won the 2024 Nobel Prize in Physics?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for future conversation threading.",
        examples=["user-abc-123", None],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"prompt": "What is (100 * 3.14) rounded?"},
                {"prompt": "Search for the latest AI news.", "session_id": "sess-001"},
            ]
        }
    }


class AgentResponse(BaseModel):
    """
    Response body returned by the POST /ask endpoint.

    Fields
    ------
    answer : str
        The final answer produced by the ReAct agent.
    status : str
        A short status string indicating the outcome.
        Values: ``"success"`` | ``"error"``
    """

    answer: str = Field(
        ...,
        description="The agent's final answer to the user's prompt.",
    )
    status: str = Field(
        ...,
        description="Outcome of the agent run: 'success' or 'error'.",
        examples=["success", "error"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"answer": "The result is 314.0", "status": "success"},
                {"answer": "Agent error: model did not return valid JSON.", "status": "error"},
            ]
        }
    }

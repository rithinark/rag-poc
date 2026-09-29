"""
FastAPI Streaming Server exposing the Secure RAG Chatbot to the chat-langchain frontend.
Streams agent reasoning, real-time tokens, and MCP tool call executions over SSE.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag_poc.agent.chatbot import create_enterprise_chatbot
from rag_poc.agent.mcp_client import get_secure_mcp_tools
from rag_poc.core.models import ClassificationLevel, JWTClaims
from rag_poc.security.jwt_auth import JWTAuthenticator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Secure RAG Chatbot Service",
    description="Enterprise LangChain + Google Gemini Agent with FastMCP Retrieval",
    version="1.0.0",
)

# Enable CORS for Next.js / Vite chat-langchain frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

authenticator = JWTAuthenticator()


class LoginRequest(BaseModel):
    user_id: str = Field(default="alice@corp.com")
    tenant_id: str = Field(default="tenant_alpha")
    security_groups: List[str] = Field(default_factory=lambda: ["security_team", "default"])
    clearance_level: ClassificationLevel = Field(default=ClassificationLevel.INTERNAL)


class LoginResponse(BaseModel):
    token: str
    tenant_id: str
    user_id: str
    clearance_level: str


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(default="default_thread")


async def get_caller_auth(
    authorization: Optional[str] = Header(None),
) -> tuple[str, JWTClaims]:
    """Validates incoming Authorization: Bearer <jwt> header."""
    if not authorization or not authorization.startswith("Bearer "):
        # For development / convenience fallback: default to tenant_alpha session
        token = authenticator.generate_token(
            sub="default_user@corp.com",
            tenant_id="tenant_alpha",
            security_groups=["security_team", "default"],
            clearance_level=ClassificationLevel.INTERNAL,
        )
        claims = authenticator.verify_token(token)
        return token, claims

    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = authenticator.verify_token(token)
        return token, claims
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired JWT token: {str(e)}",
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "secure-rag-chatbot"}


@app.post("/api/auth/token", response_model=LoginResponse)
async def generate_auth_token(req: LoginRequest):
    """
    Demo login endpoint: generates cryptographically signed JWT tokens
    for testing tenant isolation (e.g. tenant_alpha vs tenant_beta).
    """
    token = authenticator.generate_token(
        sub=req.user_id,
        tenant_id=req.tenant_id,
        security_groups=req.security_groups,
        clearance_level=req.clearance_level,
    )
    return LoginResponse(
        token=token,
        tenant_id=req.tenant_id,
        user_id=req.user_id,
        clearance_level=req.clearance_level.value,
    )


@app.post("/api/chat/stream")
async def chat_stream(
    payload: ChatMessageRequest,
    auth: tuple[str, JWTClaims] = Depends(get_caller_auth),
):
    """
    Streams LangChain agent execution events (tokens, tool calls, and citations)
    via Server-Sent Events (SSE) to the chat-langchain frontend.
    """
    token, claims = auth
    tools = await get_secure_mcp_tools(auth_token=token, tenant_id=claims.tenant_id)
    agent = create_enterprise_chatbot(tools)

    async def event_generator() -> AsyncGenerator[Dict[str, str], None]:
        config = {"configurable": {"thread_id": payload.thread_id}}
        inputs = {"messages": [("user", payload.message)]}

        try:
            async for event in agent.astream_events(inputs, config, version="v2"):
                kind = event["event"]

                # 1. LLM Token Streaming
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        text_chunk = chunk.content
                        if isinstance(text_chunk, str):
                            yield {"event": "token", "data": json.dumps({"content": text_chunk})}

                # 2. Tool Execution Start (Visualized in UI as "Querying Knowledge Base...")
                elif kind == "on_tool_start":
                    yield {
                        "event": "tool_start",
                        "data": json.dumps({
                            "tool": event["name"],
                            "input": event["data"].get("input"),
                            "status": "Searching verified enterprise knowledge base...",
                        }),
                    }

                # 3. Tool Execution Complete (Passages and Citations)
                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    yield {
                        "event": "tool_end",
                        "data": json.dumps({
                            "tool": event["name"],
                            "output": str(output),
                        }),
                    }

            yield {"event": "done", "data": json.dumps({"status": "completed"})}

        except Exception as e:
            logger.error("Error in agent stream: %s", e, exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Agent error: {str(e)}"}),
            }

    return EventSourceResponse(event_generator())


@app.post("/api/chat")
async def chat_sync(
    payload: ChatMessageRequest,
    auth: tuple[str, JWTClaims] = Depends(get_caller_auth),
):
    """Non-streaming synchronous chat endpoint."""
    token, claims = auth
    tools = await get_secure_mcp_tools(auth_token=token, tenant_id=claims.tenant_id)
    agent = create_enterprise_chatbot(tools)

    config = {"configurable": {"thread_id": payload.thread_id}}
    result = await agent.ainvoke({"messages": [("user", payload.message)]}, config)
    messages = result.get("messages", [])
    response_text = messages[-1].content if messages else "No response generated."

    return {
        "response": response_text,
        "tenant_id": claims.tenant_id,
        "thread_id": payload.thread_id,
    }


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Entrypoint to launch the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()

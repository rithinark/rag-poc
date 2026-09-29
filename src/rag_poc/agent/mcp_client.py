"""
MCP Tool Client & Security Context Injector for LangChain.
Connects to FastMCP Server via SSE, loads dynamic tools, and securely binds
caller identity (auth_token & tenant_id) to prevent cross-tenant impersonation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from rag_poc.core.config import settings

logger = logging.getLogger(__name__)


class ScopedRetrievalInput(BaseModel):
    query: str = Field(..., description="Semantic search query expressing the information need")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve")


def _format_passages_to_xml(raw_result: Any) -> str:
    """Converts any MCP or vector tool output into standard <context> XML tags."""
    if isinstance(raw_result, str):
        if "<context" in raw_result:
            return raw_result
        try:
            raw_result = json.loads(raw_result)
        except Exception:
            if raw_result.strip():
                return (
                    '<context id="doc_context" source="internal-docs" classification="INTERNAL" data_only="true">\n'
                    f"{raw_result.strip()}\n"
                    "</context>"
                )
            return "No matching verified enterprise passages found."

    # If raw_result has content attribute (e.g. CallToolResult or AIMessage)
    if hasattr(raw_result, "content") and not isinstance(raw_result, dict):
        raw_result = raw_result.content

    # Unpack list or single object
    candidates = raw_result if isinstance(raw_result, list) else [raw_result]
    items = []

    for entry in candidates:
        text_payload = None
        if isinstance(entry, dict) and "text" in entry and isinstance(entry["text"], str):
            text_payload = entry["text"]
        elif hasattr(entry, "text") and isinstance(entry.text, str):
            text_payload = entry.text

        if text_payload is not None:
            try:
                parsed = json.loads(text_payload)
                if isinstance(parsed, list):
                    items.extend(parsed)
                elif isinstance(parsed, dict):
                    items.append(parsed)
                else:
                    items.append(str(parsed))
                continue
            except Exception:
                items.append(text_payload)
                continue

        items.append(entry)

    formatted = []
    for item in items:
        if isinstance(item, dict):
            # In case dict has wrapped "text" without "content"
            if "text" in item and "content" not in item and isinstance(item["text"], str):
                try:
                    inner = json.loads(item["text"])
                    if isinstance(inner, dict):
                        item = inner
                except Exception:
                    pass

            cid = item.get("chunk_id") or item.get("document_id") or "chunk_0"
            s_uri = item.get("source_uri") or item.get("document_id") or "internal-docs"
            cls_val = item.get("classification", "INTERNAL")
            if isinstance(cls_val, dict):
                cls_val = cls_val.get("value", "INTERNAL")
            elif hasattr(cls_val, "value"):
                cls_val = cls_val.value
            c_text = item.get("content") or item.get("text", "")
            if c_text and str(c_text).strip():
                formatted.append(
                    f'<context id="{cid}" source="{s_uri}" classification="{cls_val}" data_only="true">\n'
                    f"{str(c_text).strip()}\n"
                    f"</context>"
                )
        elif hasattr(item, "content"):
            cid = getattr(item, "chunk_id", "") or getattr(item, "document_id", "chunk_0")
            s_uri = getattr(item, "source_uri", "") or getattr(item, "document_id", "internal-docs")
            cls_val = getattr(item, "classification", "INTERNAL")
            if hasattr(cls_val, "value"):
                cls_val = cls_val.value
            c_text = getattr(item, "content", "")
            if c_text and str(c_text).strip():
                formatted.append(
                    f'<context id="{cid}" source="{s_uri}" classification="{cls_val}" data_only="true">\n'
                    f"{str(c_text).strip()}\n"
                    f"</context>"
                )
        elif isinstance(item, str) and item.strip():
            if "<context" in item:
                formatted.append(item.strip())
            else:
                formatted.append(
                    f'<context id="doc_context" source="internal-docs" classification="INTERNAL" data_only="true">\n'
                    f"{item.strip()}\n"
                    f"</context>"
                )

    if formatted:
        return "\n\n".join(formatted)
    return "No matching verified enterprise passages found."


async def get_secure_mcp_tools(auth_token: str, tenant_id: str) -> List[BaseTool]:
    """
    Connects to the FastMCP SSE endpoint, fetches tool definitions, and wraps
    `query_knowledge_base` so that the authenticated caller's `tenant_id` and `auth_token`
    are enforced server-side.
    """
    mcp_url = f"http://{settings.mcp_host}:{settings.mcp_port}/sse"
    if settings.mcp_host == "0.0.0.0":
        mcp_url = f"http://127.0.0.1:{settings.mcp_port}/sse"

    client = MultiServerMCPClient(
        {
            "rag_service": {
                "url": mcp_url,
                "transport": "sse",
            }
        }
    )

    try:
        raw_tools = await client.get_tools()
    except Exception as e:
        logger.warning(
            "Could not connect to FastMCP SSE endpoint at %s: %s. Using local tool binding.",
            mcp_url,
            e,
        )
        from rag_poc.mcp.tools import query_knowledge_base

        async def _local_scoped_retrieval(query: str, top_k: int = 5) -> str:
            passages = await query_knowledge_base(
                query=query,
                tenant_id=tenant_id,
                top_k=top_k,
                auth_token=auth_token,
            )
            return _format_passages_to_xml(passages)

        return [
            StructuredTool.from_function(
                coroutine=_local_scoped_retrieval,
                name="query_knowledge_base",
                description=(
                    "Search authoritative enterprise knowledge base for verified internal facts, "
                    "technical standards, and company policies."
                ),
                args_schema=ScopedRetrievalInput,
            )
        ]

    wrapped_tools: List[BaseTool] = []
    for tool in raw_tools:
        if tool.name == "query_knowledge_base":
            async def _scoped_retrieval(query: str, top_k: int = 5) -> str:
                raw_result = await tool.ainvoke({
                    "query": query,
                    "tenant_id": tenant_id,
                    "top_k": top_k,
                    "confidence_threshold": 0.0,
                    "auth_token": auth_token,
                })
                return _format_passages_to_xml(raw_result)

            wrapped_tools.append(
                StructuredTool.from_function(
                    coroutine=_scoped_retrieval,
                    name="query_knowledge_base",
                    description=(
                        "Search authoritative enterprise knowledge base for verified internal facts, "
                        "technical standards, and company policies."
                    ),
                    args_schema=ScopedRetrievalInput,
                )
            )
        else:
            wrapped_tools.append(tool)

    return wrapped_tools

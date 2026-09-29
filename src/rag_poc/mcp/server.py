"""
FastMCP Server exposing the enterprise retrieval tool over SSE (HTTP) and stdio transports.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rag_poc.core.config import settings
from rag_poc.core.models import RetrievedPassage
from rag_poc.mcp.tools import query_knowledge_base

logger = logging.getLogger(__name__)


def create_mcp_server() -> FastMCP:
    """Creates and configures the enterprise FastMCP server instance."""
    mcp = FastMCP(
        name="enterprise-rag-retrieval-service",
        instructions="Authoritative retrieval tool providing verified internal documentation context with pre-retrieval security boundaries.",
        host=settings.mcp_host,
        port=settings.mcp_port,
    )

    @mcp.tool(
        name="query_knowledge_base",
        description="Search enterprise knowledge base for grounded factual context based on a natural language query with strict security scoping."
    )
    async def _query_knowledge_base(
        query: str = Field(..., description="Semantic search query string expressing information need"),
        tenant_id: str = Field(..., description="Access scope and security boundary identifier"),
        top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve"),
        confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff"),
        auth_token: Optional[str] = Field(default=None, description="Signed JWT bearer token for caller identity validation"),
    ) -> List[RetrievedPassage]:
        return await query_knowledge_base(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            auth_token=auth_token,
        )

    return mcp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting FastMCP Retrieval Service on {settings.mcp_host}:{settings.mcp_port} (transport: {settings.mcp_transport})")
    server = create_mcp_server()
    server.run(transport=settings.mcp_transport)

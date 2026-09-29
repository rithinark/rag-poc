"""
Model Context Protocol (MCP) Integration Package.
Exposes secure, pre-filtered retrieval as an agentic MCP Tool.
"""

from .tools import query_knowledge_base, set_active_service
from .server import create_mcp_server

__all__ = ["query_knowledge_base", "set_active_service", "create_mcp_server"]

"""
Application configuration and security policy parameters for refactored RAG architecture.
"""

from __future__ import annotations

import os
from typing import Optional
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

# Automatically discover and load .env file from workspace root
load_dotenv(find_dotenv(usecwd=True))


def _get_env_secret(key: str, fallback_key: Optional[str] = None) -> Optional[str]:
    for k in (key, fallback_key):
        if k:
            val = os.getenv(k)
            if val and val.strip():
                return val.strip()
    return None


class AppConfig(BaseModel):
    # Google Gemini / LLM Settings
    gemini_api_key: Optional[str] = Field(default_factory=lambda: _get_env_secret("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    google_api_key: Optional[str] = Field(default_factory=lambda: _get_env_secret("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))

    # Qdrant Vector Database Settings
    qdrant_location: str = Field(default_factory=lambda: os.getenv("QDRANT_LOCATION", ":memory:"))
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", ":memory:"))
    qdrant_collection: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "secure_knowledge_base"))
    qdrant_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", None))

    # FastEmbed & Embedding Provider Settings
    fastembed_model: str = Field(default_factory=lambda: os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_model_name: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"))
    embedding_provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "fastembed"))
    embedding_dim: int = Field(default=384)
    openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_base: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_BASE", None))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", None))
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # Cryptographic HMAC Tamper-Proof Signing
    hmac_secret_key: str = Field(default_factory=lambda: os.getenv("HMAC_SECRET_KEY", "super-secure-production-hmac-key-256bit!!"))

    # Meta Prompt Guard ONNX Settings
    prompt_guard_model_id: str = Field(default="meta-llama/Prompt-Guard-86M")
    prompt_guard_model_path: str = Field(default_factory=lambda: os.getenv("PROMPT_GUARD_MODEL_PATH", "data/models/prompt_guard.onnx"))
    prompt_guard_onnx_path: Optional[str] = Field(default_factory=lambda: os.getenv("PROMPT_GUARD_ONNX_PATH", None))
    ipi_threshold: float = Field(default=0.75, description="Classifier threshold above which retrieved chunks are dropped")

    # FastMCP Protocol Settings
    mcp_host: str = Field(default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"))
    mcp_port: int = Field(default_factory=lambda: int(os.getenv("MCP_PORT", "8001")))
    mcp_transport: str = Field(default_factory=lambda: os.getenv("MCP_TRANSPORT", "sse"))

    # JWT Authentication
    jwt_secret_key: str = Field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", "super-secret-enterprise-hmac-key-256bit!!"))
    jwt_algorithm: str = Field(default="HS256")
    jwt_issuer: str = Field(default="enterprise-identity-broker")
    jwt_audience: str = Field(default="enterprise-rag-service")

    # Retrieval Defaults
    default_top_k: int = Field(default=5)
    default_confidence_threshold: float = Field(default=0.60)


settings = AppConfig()

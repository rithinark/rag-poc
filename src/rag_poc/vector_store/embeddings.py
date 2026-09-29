"""
Embedding Providers with ONNX Runtime Acceleration and OpenAI Swappability.
Supports FastEmbed (BAAI/bge-small-en-v1.5, 384-dim ONNX) natively and OpenAI-compatible providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import httpx
from fastembed import TextEmbedding

from rag_poc.core.config import settings


class BaseEmbeddingProvider(ABC):
    """Abstract contract for text embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated embedding vectors."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of strings."""
        pass


class FastEmbedProvider(BaseEmbeddingProvider):
    """
    Local ONNX-accelerated embedding provider using FastEmbed.
    Defaults to BAAI/bge-small-en-v1.5 (384 dimensions).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.fastembed_model
        self._model = TextEmbedding(model_name=self.model_name)
        self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider connecting to any OpenAI-compatible /v1/embeddings endpoint
    (OpenAI, vLLM, Ollama, LM Studio, etc.).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: int = 1536,
    ):
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key or "sk-dummy"
        self.model_name = model_name or settings.openai_embedding_model
        self._dimension = dimension
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.post(
            f"{self.base_url}/embeddings",
            json={"input": texts, "model": self.model_name},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings


def get_default_embedding_provider() -> BaseEmbeddingProvider:
    """Factory creating the configured embedding provider."""
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbeddingProvider()
    return FastEmbedProvider()

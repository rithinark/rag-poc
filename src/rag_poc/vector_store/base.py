"""
Vector store abstract interface defining standard upsert and filtered search contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from rag_poc.core.models import Chunk, RetrievalFilter, RetrievedPassage


class BaseVectorStore(ABC):
    """Abstract Vector Store contract."""

    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> int:
        """Inserts or updates chunks and their embeddings into the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        security_filter: Optional[RetrievalFilter] = None,
        confidence_threshold: float = 0.0,
    ) -> List[RetrievedPassage]:
        """
        Executes pre-retrieval filtered approximate nearest neighbor (ANN) search.
        Evaluates security boundaries directly within index traversal.
        """
        pass

    @abstractmethod
    def search_unfiltered(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievedPassage]:
        """
        Executes un-scoped nearest neighbor search across the entire corpus.
        Used exclusively for comparative evaluation and demonstrating Context Starvation.
        """
        pass

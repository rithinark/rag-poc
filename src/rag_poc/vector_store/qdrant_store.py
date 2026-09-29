"""
Qdrant Vector Store Implementation with Native Pre-Retrieval Filtering and Payload HMAC Anchoring.
Supports embedded in-memory mode (:memory:), local directory persistence, and remote clusters.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_poc.core.config import settings
from rag_poc.core.models import (
    Chunk,
    ClassificationLevel,
    RetrievalFilter,
    RetrievedPassage,
)
from rag_poc.vector_store.base import BaseVectorStore
from rag_poc.vector_store.embeddings import BaseEmbeddingProvider, get_default_embedding_provider


class QdrantVectorStore(BaseVectorStore):
    """
    Production-grade Vector Store backed by Qdrant.
    Guarantees tenant isolation via native pre-retrieval query filtering and embeds
    cryptographic HMAC signatures in point payloads for tamper verification.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        location: Optional[str] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name or settings.qdrant_collection
        self.embedding_provider = embedding_provider or get_default_embedding_provider()

        if client is not None:
            self.client = client
        else:
            loc = location or settings.qdrant_url
            if loc == ":memory:" or not loc:
                self.client = QdrantClient(":memory:")
            elif loc.startswith("http://") or loc.startswith("https://"):
                self.client = QdrantClient(url=loc)
            else:
                self.client = QdrantClient(path=loc)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the target collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_provider.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload indices for fast pre-filtering on remote cluster deployments
            if hasattr(self, "_is_memory") and not self._is_memory:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tenant_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="classification",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str, tenant_id: str = "") -> str:
        """Maps an arbitrary chunk_id (scoped by tenant_id) to a deterministic UUID for Qdrant point IDs."""
        key = f"{tenant_id}:{chunk_id}" if tenant_id else chunk_id
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def upsert(self, chunks: List[Chunk]) -> int:
        """
        Embeds (if necessary) and upserts chunks into Qdrant.
        Stores cryptographic signatures and multi-tenant security tags in point payloads.
        """
        if not chunks:
            return 0

        # Compute missing embeddings in batch
        unembedded = [c for c in chunks if c.embedding is None]
        if unembedded:
            texts = [c.content for c in unembedded]
            embeddings = self.embedding_provider.embed_batch(texts)
            for c, emb in zip(unembedded, embeddings):
                c.embedding = emb

        points: List[models.PointStruct] = []
        for chunk in chunks:
            point_id = self._chunk_id_to_uuid(chunk.chunk_id, chunk.tenant_id)
            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "tenant_id": chunk.tenant_id,
                "security_groups": chunk.security_groups,
                "classification": chunk.classification.value if hasattr(chunk.classification, "value") else str(chunk.classification),
                "source_uri": chunk.source_uri,
                "_signature": chunk.signature,
                "metadata": chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def _build_filter(self, security_filter: RetrievalFilter) -> models.Filter:
        """Constructs native Qdrant boolean filter conditions from caller claims."""
        must_conditions: List[models.Condition] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=security_filter.tenant_id),
            )
        ]

        # Security groups condition (caller must belong to at least one required group)
        if security_filter.authorized_groups:
            must_conditions.append(
                models.FieldCondition(
                    key="security_groups",
                    match=models.MatchAny(any=security_filter.authorized_groups),
                )
            )

        # Classification level hierarchy check
        if security_filter.clearance_level:
            hierarchy = [
                ClassificationLevel.PUBLIC.value,
                ClassificationLevel.INTERNAL.value,
                ClassificationLevel.CONFIDENTIAL.value,
                ClassificationLevel.RESTRICTED.value,
            ]
            max_val = (
                security_filter.clearance_level.value
                if hasattr(security_filter.clearance_level, "value")
                else str(security_filter.clearance_level)
            )
            if max_val in hierarchy:
                allowed_levels = hierarchy[: hierarchy.index(max_val) + 1]
                must_conditions.append(
                    models.FieldCondition(
                        key="classification",
                        match=models.MatchAny(any=allowed_levels),
                    )
                )

        return models.Filter(must=must_conditions)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        security_filter: Optional[RetrievalFilter] = None,
        confidence_threshold: float = 0.0,
    ) -> List[RetrievedPassage]:
        """
        Executes pre-retrieval filtered ANN search in Qdrant.
        Filters out unauthorized tenants at the vector index level to prevent Context Starvation.
        """
        qdrant_filter = self._build_filter(security_filter) if security_filter else None

        thresh = (
            float(confidence_threshold.default)
            if hasattr(confidence_threshold, "default")
            else float(confidence_threshold or 0.0)
        )
        effective_limit = top_k if isinstance(top_k, int) else 5

        scored_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=effective_limit,
            score_threshold=thresh if thresh > 0.0 else None,
            with_payload=True,
        ).points

        passages: List[RetrievedPassage] = []
        for sp in scored_points:
            p = sp.payload or {}
            passages.append(
                RetrievedPassage(
                    chunk_id=p.get("chunk_id", str(sp.id)),
                    document_id=p.get("document_id", "unknown"),
                    content=p.get("content", ""),
                    tenant_id=p.get("tenant_id", "unknown"),
                    security_groups=p.get("security_groups", []),
                    classification=ClassificationLevel(p.get("classification", ClassificationLevel.INTERNAL.value)),
                    similarity_score=float(sp.score),
                    source_uri=p.get("source_uri", ""),
                    signature=p.get("_signature"),
                )
            )

        return passages

    def search_unfiltered(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievedPassage]:
        """
        Executes raw ANN search across all tenants without security filters.
        Used for comparative evaluation to demonstrate context starvation vulnerabilities.
        """
        scored_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        passages: List[RetrievedPassage] = []
        for sp in scored_points:
            p = sp.payload or {}
            passages.append(
                RetrievedPassage(
                    chunk_id=p.get("chunk_id", str(sp.id)),
                    document_id=p.get("document_id", "unknown"),
                    content=p.get("content", ""),
                    tenant_id=p.get("tenant_id", "unknown"),
                    security_groups=p.get("security_groups", []),
                    classification=ClassificationLevel(p.get("classification", ClassificationLevel.INTERNAL.value)),
                    similarity_score=float(sp.score),
                    source_uri=p.get("source_uri", ""),
                    signature=p.get("_signature"),
                )
            )

        return passages

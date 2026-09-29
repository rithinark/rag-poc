"""
Model Context Protocol Tools for Secure Knowledge Base Retrieval.
Implements execution rails, input prompt guard / PII screening,
Qdrant pre-retrieval tenant filtering, and HMAC/IPI retrieval auditing.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from pydantic import Field

from rag_poc.core.models import ClassificationLevel, Document, RetrievalFilter, RetrievedPassage
from rag_poc.guardrails.input_rails import AsyncInputGuardrailPipeline
from rag_poc.guardrails.retrieval_rails import AsyncRetrievalGuardrailPipeline
from rag_poc.ingestion.chunker import TiktokenRecursiveChunker
from rag_poc.ingestion.hf_dataset import get_customer_support_documents
from rag_poc.ingestion.signer import HMACChunkSigner
from rag_poc.ingestion.workplace_dataset import get_workplace_documents
from rag_poc.security.broker import SecurityBroker
from rag_poc.security.jwt_auth import AuthenticationError, JWTAuthenticator
from rag_poc.vector_store.base import BaseVectorStore
from rag_poc.vector_store.embeddings import BaseEmbeddingProvider, get_default_embedding_provider
from rag_poc.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def _seed_default_knowledge_base(vector_store: BaseVectorStore, signer: HMACChunkSigner) -> None:
    """Seeds foundational enterprise documentation for tenant_alpha and tenant_beta if store is empty."""
    try:
        if hasattr(vector_store, "client") and hasattr(vector_store, "collection_name"):
            count = vector_store.client.count(vector_store.collection_name).count
            if count > 0:
                return
    except Exception:
        pass

    chunker = TiktokenRecursiveChunker(chunk_size_tokens=256, chunk_overlap_tokens=32)

    docs = [
        Document(
            document_id="doc_corp_sec_01",
            title="Enterprise Key Management Standards",
            source_uri="doc://security/key-management-standards.md",
            content=(
                "# Enterprise Key Management Standards\n"
                "All AES-256 cryptographic master keys and encryption keys must be rotated every 90 days in compliance with enterprise security policy.\n"
                "Hardware security modules (HSM) must comply with FIPS 140-3 Level 3.\n\n"
                "## Rotation Schedule\n"
                "- DEK (Data Encryption Key): Rotated every 30 days in memory.\n"
                "- KEK (Key Encryption Key): Rotated every 90 days in HSM.\n"
                "- Master / Root Keys: Rotated every 365 days in an air-gapped vault.\n\n"
                "Access to master encryption keys is strictly restricted to Cloud Security Officers."
            ),
            tenant_id="tenant_alpha",
            security_groups=["security_team", "default"],
            classification=ClassificationLevel.INTERNAL,
        ),
        Document(
            document_id="doc_corp_sec_02",
            title="KMS Cryptographic Backup Master Key Storage",
            source_uri="doc://security/kms-backup-storage.md",
            content=(
                "# KMS Cryptographic Backup Storage Policy\n"
                "KMS cryptographic backup master keys are stored in an encrypted offline air-gapped vault.\n"
                "Access requires multi-party quorum authorization (M-of-N Shamir secret sharing requiring 3 of 5 executive key holders).\n"
                "Backup media is verified quarterly against tamper-evident physical and cryptographic seals."
            ),
            tenant_id="tenant_alpha",
            security_groups=["security_team", "default"],
            classification=ClassificationLevel.INTERNAL,
        ),
        Document(
            document_id="doc_beta_01",
            title="Tenant Beta Core Engineering Architecture Blueprints",
            source_uri="doc://engineering/architecture-blueprints.md",
            content=(
                "# Tenant Beta Engineering Architecture\n"
                "Core banking integration services use zero-trust mTLS service mesh on private Kubernetes clusters.\n"
                "Q4 marketing and infrastructure allocation is $500,000 across targeted enterprise expansion."
            ),
            tenant_id="tenant_beta",
            security_groups=["core_engineering", "default"],
            classification=ClassificationLevel.CONFIDENTIAL,
        ),
    ]

    # Add relatable Workplace & Employee Handbook documents
    docs.extend(get_workplace_documents(tenant_id="tenant_alpha"))
    docs.extend(get_workplace_documents(tenant_id="tenant_beta"))

    # Add Hugging Face Customer Support & E-Commerce documents
    docs.extend(get_customer_support_documents(tenant_id="tenant_alpha"))
    docs.extend(get_customer_support_documents(tenant_id="tenant_beta"))

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))

    signed_chunks = signer.sign_chunks(all_chunks)
    vector_store.upsert(signed_chunks)
    logger.info("Seeded %d foundational enterprise document chunks into vector store.", len(signed_chunks))


class RetrievalServiceContext:
    """Holds active state for MCP retrieval execution."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        signer: Optional[HMACChunkSigner] = None,
        input_rails: Optional[AsyncInputGuardrailPipeline] = None,
        retrieval_rails: Optional[AsyncRetrievalGuardrailPipeline] = None,
        broker: Optional[SecurityBroker] = None,
        authenticator: Optional[JWTAuthenticator] = None,
    ):
        self.embedding_provider = embedding_provider or get_default_embedding_provider()
        self.vector_store = vector_store or QdrantVectorStore(embedding_provider=self.embedding_provider)
        self.signer = signer or HMACChunkSigner()
        self.input_rails = input_rails or AsyncInputGuardrailPipeline()
        self.retrieval_rails = retrieval_rails or AsyncRetrievalGuardrailPipeline(signer=self.signer)
        self.broker = broker or SecurityBroker()
        self.authenticator = authenticator or JWTAuthenticator()
        _seed_default_knowledge_base(self.vector_store, self.signer)


# Global execution context
_active_context = RetrievalServiceContext()


def set_active_service(context: RetrievalServiceContext) -> None:
    """Sets the active service context for MCP tools."""
    global _active_context
    _active_context = context


def get_active_service() -> RetrievalServiceContext:
    """Returns the current active service context."""
    global _active_context
    return _active_context


async def query_knowledge_base(
    query: str = Field(..., description="Semantic search query string expressing information need"),
    tenant_id: str = Field(..., description="Access scope and security boundary identifier"),
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve"),
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity cutoff"),
    auth_token: Optional[str] = Field(default=None, description="Cryptographically signed OAuth2/OIDC JWT bearer token"),
) -> List[RetrievedPassage]:
    """
    Executes scoped vector search against Qdrant with:
    1. Input Guardrails: Prompt Guard ONNX screen + Presidio PII anonymization.
    2. Execution Rails: Identity enforcement & JWT claim compilation into native Qdrant pre-filter.
    3. Retrieval Rails: Cryptographic HMAC tamper verification + IPI Drop & Quarantine defense.
    """
    ctx = get_active_service()

    # 1. INPUT RAILS: Meta Prompt Guard screen & Presidio PII Anonymization
    input_verdict = ctx.input_rails.process(query)
    if not input_verdict.is_safe:
        raise PermissionError(input_verdict.rejection_reason)

    effective_query = input_verdict.sanitized_query

    # 2. EXECUTION RAIL: Identity Enforcement & Anti-Impersonation
    if auth_token:
        claims = ctx.authenticator.verify_token(auth_token)
        # Anti-Impersonation: Prevent caller from requesting a foreign tenant scope
        if claims.tenant_id != tenant_id:
            raise PermissionError(
                f"Security Rail Violation: Caller token tenant '{claims.tenant_id}' does not match requested tenant '{tenant_id}'"
            )
        security_filter = ctx.broker.claims_to_filter(claims)
    else:
        # Fallback to strict single-tenant filter without privileged group clearance
        security_filter = RetrievalFilter(
            tenant_id=tenant_id,
            authorized_groups=["default"],
            clearance_level=ClassificationLevel.INTERNAL,
        )

    # 3. Vector Embedding Generation
    query_vec = ctx.embedding_provider.embed_text(effective_query)

    # 4. Pre-Retrieval Scoped Vector Search in Qdrant
    limit = top_k if isinstance(top_k, int) else (int(top_k.default) if hasattr(top_k, "default") else 5)
    cutoff = float(confidence_threshold.default) if hasattr(confidence_threshold, "default") else float(confidence_threshold or 0.0)

    candidate_passages = ctx.vector_store.search(
        query_embedding=query_vec,
        top_k=limit,
        security_filter=security_filter,
        confidence_threshold=cutoff,
    )

    # 5. RETRIEVAL RAILS: HMAC Tamper Verification & IPI Drop & Quarantine
    rail_result = ctx.retrieval_rails.process(candidate_passages)

    if rail_result.tampered_count > 0:
        logger.warning(
            "Retrieval Rail Alert: Dropped %d tampered or un-ledgered chunk(s) from results!",
            rail_result.tampered_count,
        )

    if rail_result.quarantined_chunks:
        logger.warning(
            "Retrieval Rail Alert: Quarantined %d indirect prompt injection chunk(s) from results!",
            len(rail_result.quarantined_chunks),
        )

    verified = rail_result.verified_passages
    if verified and len(verified) > 1:
        top_score = verified[0].similarity_score
        if top_score > 0.0:
            # Dynamic relevance threshold: drop tail passages below 65% of top similarity to boost Context Precision
            min_score = top_score * 0.65
            verified = [p for p in verified if p.similarity_score >= min_score]

    return verified

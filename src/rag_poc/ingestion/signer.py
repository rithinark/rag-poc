"""
HMAC-SHA256 Cryptographic Chunk Signer & Tamper-Proof Verifier.
Anchors tamper-proof cryptographic signatures directly within vector payloads.
"""

from __future__ import annotations

import hmac
import hashlib
from typing import List, Optional

from rag_poc.core.config import settings
from rag_poc.core.models import Chunk, RetrievedPassage


class HMACChunkSigner:
    """
    Signs chunks at ingestion and validates signatures during retrieval to guarantee
    authenticity and detect vector store index tampering (PoisonedRAG).
    """

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = (secret_key or settings.hmac_secret_key).encode("utf-8")

    def compute_signature(self, chunk_id: str, content: str, tenant_id: str) -> str:
        """Computes HMAC-SHA256 signature for a chunk's core attributes."""
        canonical_payload = f"{chunk_id}:{tenant_id}:{content}".encode("utf-8")
        return hmac.new(self.secret_key, canonical_payload, hashlib.sha256).hexdigest()

    def sign_chunk(self, chunk: Chunk) -> Chunk:
        """Generates and sets the HMAC-SHA256 signature anchor on a chunk."""
        chunk.signature = self.compute_signature(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            tenant_id=chunk.tenant_id,
        )
        return chunk

    def sign_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Signs a batch of chunks."""
        for chunk in chunks:
            self.sign_chunk(chunk)
        return chunks

    def verify_chunk(self, chunk_id: str, content: str, tenant_id: str, signature: Optional[str]) -> bool:
        """Verifies that the provided signature matches the expected HMAC."""
        if not signature:
            return False
        expected = self.compute_signature(chunk_id=chunk_id, content=content, tenant_id=tenant_id)
        return hmac.compare_digest(expected, signature)

    def verify_passage(self, passage: RetrievedPassage) -> bool:
        """Verifies the cryptographic signature of a retrieved passage."""
        return self.verify_chunk(
            chunk_id=passage.chunk_id,
            content=passage.content,
            tenant_id=passage.tenant_id,
            signature=passage.signature,
        )

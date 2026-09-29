"""
Structure-Aware Text Chunker using LangChain RecursiveCharacterTextSplitter and tiktoken.
"""

from __future__ import annotations

import uuid
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_poc.core.models import Chunk, Document


class TiktokenRecursiveChunker:
    """
    Decomposes extracted document text into semantic chunks using LangChain's
    RecursiveCharacterTextSplitter with tiktoken token counting.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 256,
        chunk_overlap_tokens: int = 32,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size_tokens
        self.chunk_overlap = chunk_overlap_tokens
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "|", ". ", " ", ""],
        )

    def split_text(self, text: str) -> List[str]:
        """Splits raw or markdown text into token-bounded passages."""
        return self.splitter.split_text(text)

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Splits a Document into typed Chunks with preserved metadata."""
        split_texts = self.split_text(doc.content)
        chunks: List[Chunk] = []

        for idx, text in enumerate(split_texts):
            chunk_id = f"{doc.document_id}_chk_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    content=text.strip(),
                    tenant_id=doc.tenant_id,
                    security_groups=doc.security_groups,
                    classification=doc.classification,
                    source_uri=doc.source_uri,
                    metadata={
                        "title": doc.title,
                        "chunk_index": idx,
                        "total_chunks": len(split_texts),
                        **doc.metadata,
                    },
                )
            )

        return chunks

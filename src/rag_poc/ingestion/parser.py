"""
Document Layout Parser using Docling.
Extracts rich structured elements (markdown, tables, hierarchies) from documents
without brittle regex heuristics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from docling.document_converter import DocumentConverter

from rag_poc.core.models import ClassificationLevel, Document


class ExtractedElement(BaseModel):
    element_type: str = Field(..., description="header, paragraph, table, list, text")
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document_id: str
    title: str
    tenant_id: str
    security_groups: List[str]
    classification: ClassificationLevel
    source_uri: str
    content: str
    elements: List[ExtractedElement]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DoclingDocumentParser:
    """
    Parses complex unstructured documents (PDF, DOCX, HTML, Markdown, TXT) into
    structured representations using Docling.
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def parse_file(
        self,
        file_path: str | Path,
        doc_id: Optional[str] = None,
        title: Optional[str] = None,
        tenant_id: str = "default",
        security_groups: Optional[List[str]] = None,
        classification: ClassificationLevel = ClassificationLevel.INTERNAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        """Parses a local document file using Docling layout parsing."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        conv_res = self.converter.convert(path)
        markdown_text = conv_res.document.export_to_markdown()

        elements: List[ExtractedElement] = []

        # Extract structured tables
        for idx, table in enumerate(conv_res.document.tables):
            table_md = table.export_to_markdown() if hasattr(table, "export_to_markdown") else str(table)
            elements.append(
                ExtractedElement(
                    element_type="table",
                    text=table_md,
                    metadata={"table_index": idx},
                )
            )

        elements.append(
            ExtractedElement(
                element_type="text",
                text=markdown_text,
                metadata={"format": path.suffix.lstrip(".")},
            )
        )

        return ParsedDocument(
            document_id=doc_id or path.stem,
            title=title or path.name,
            tenant_id=tenant_id,
            security_groups=security_groups or ["general"],
            classification=classification,
            source_uri=str(path.absolute()),
            content=markdown_text,
            elements=elements,
            metadata=metadata or {},
        )

    def parse(self, doc: Document) -> ParsedDocument:
        """
        Parses an in-memory Document object by writing to a temporary file
        and processing through Docling's converter.
        """
        suffix = ".md"
        if doc.source_uri and "." in doc.source_uri:
            ext = "." + doc.source_uri.split(".")[-1]
            if ext in [".pdf", ".docx", ".html", ".md", ".txt"]:
                suffix = ext

        with tempfile.NamedTemporaryFile("w+", suffix=suffix, delete=False, encoding="utf-8") as tmp:
            tmp.write(doc.content)
            tmp_path = Path(tmp.name)

        try:
            parsed = self.parse_file(
                file_path=tmp_path,
                doc_id=doc.document_id,
                title=doc.title,
                tenant_id=doc.tenant_id,
                security_groups=doc.security_groups,
                classification=doc.classification,
                metadata=doc.metadata,
            )
            parsed.source_uri = doc.source_uri
            return parsed
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

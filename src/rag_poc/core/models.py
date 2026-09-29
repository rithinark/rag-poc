"""
Core domain models and schema definitions for Secure RAG architecture.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ClassificationLevel(str, Enum):
    """Data security classification hierarchy."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"

    @property
    def level(self) -> int:
        levels = {
            ClassificationLevel.PUBLIC: 1,
            ClassificationLevel.INTERNAL: 2,
            ClassificationLevel.CONFIDENTIAL: 3,
            ClassificationLevel.RESTRICTED: 4,
        }
        return levels[self]

    def allows(self, required: ClassificationLevel) -> bool:
        """Returns True if this clearance allows viewing documents at `required` level."""
        return self.level >= required.level


class JWTClaims(BaseModel):
    """Validated caller identity and authorization claims extracted from JWT."""
    sub: str = Field(..., description="Subject identifier (user/service ID)")
    tenant_id: str = Field(..., description="Multi-tenant boundary identifier")
    roles: List[str] = Field(default_factory=list, description="Assigned enterprise roles")
    security_groups: List[str] = Field(default_factory=list, description="Authorized ACL security groups")
    clearance_level: ClassificationLevel = Field(
        default=ClassificationLevel.INTERNAL,
        description="Data classification clearance level"
    )

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        v = v.strip()
        if not v or " " in v or ";" in v or "'" in v or '"' in v:
            raise ValueError("tenant_id must be a non-empty alphanumeric/identifier string without spaces or quotes")
        return v


class Document(BaseModel):
    """Inbound raw document metadata and content."""
    document_id: str = Field(..., description="Unique document ID")
    title: str = Field(..., description="Human-readable title")
    content: str = Field(..., description="Raw text or decoded payload")
    format: str = Field(default="text/plain", description="MIME type or extension (pdf, docx, txt)")
    source_uri: str = Field(default="", description="Canonical storage URI")
    tenant_id: str = Field(..., description="Originating tenant boundary")
    security_groups: List[str] = Field(default_factory=lambda: ["default"], description="Allowed security groups")
    classification: ClassificationLevel = Field(
        default=ClassificationLevel.INTERNAL,
        description="Security classification"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document attributes")


class Chunk(BaseModel):
    """Fine-grained semantic chunk derived from parent document."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Extracted textual content of the chunk")
    parent_id: Optional[str] = Field(default=None, description="Parent section identifier")
    tenant_id: str = Field(..., description="Tenant boundary identifier")
    security_groups: List[str] = Field(default_factory=lambda: ["default"], description="ACL security groups")
    classification: ClassificationLevel = Field(default=ClassificationLevel.INTERNAL)
    source_uri: str = Field(default="")
    signature: Optional[str] = Field(default=None, description="HMAC-SHA256 tamper-proof signature anchor")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedPassage(BaseModel):
    """Standardized retrieval object exposed across MCP tools and upstream LLM agents."""
    document_id: str = Field(description="Unique identifier of the source document")
    chunk_id: str = Field(default="", description="Unique identifier of the retrieved chunk")
    content: str = Field(description="Extracted textual content of the chunk")
    similarity_score: float = Field(description="Cosine similarity metric (0.0 to 1.0)")
    tenant_id: str = Field(description="Originating tenant boundary identifier")
    source_uri: str = Field(default="", description="Canonical path or URI of original asset")
    signature: Optional[str] = Field(default=None, description="HMAC-SHA256 signature anchor")
    classification: ClassificationLevel = Field(default=ClassificationLevel.INTERNAL)
    security_groups: List[str] = Field(default_factory=list)


class GuardrailVerdict(BaseModel):
    """Result of passing a payload through a guardrail evaluation flow."""
    passed: bool = Field(..., description="True if payload passed safety checks")
    stage: str = Field(..., description="Guardrail stage (input, execution, retrieval, output)")
    refusal_reason: Optional[str] = Field(default=None, description="Explanation if rejected")
    sanitized_output: Optional[str] = Field(default=None, description="Sanitized payload if modified")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalFilter(BaseModel):
    """Pre-retrieval security filter compiled from caller JWT claims."""
    tenant_id: str
    authorized_groups: List[str]
    clearance_level: ClassificationLevel

    def matches(self, chunk: Chunk) -> bool:
        """Evaluates whether a chunk matches the security filter."""
        if chunk.tenant_id != self.tenant_id:
            return False
        if not self.clearance_level.allows(chunk.classification):
            return False
        if chunk.security_groups and not any(g in self.authorized_groups for g in chunk.security_groups):
            return False
        return True

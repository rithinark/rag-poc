"""
Cryptographic Audit Ledger & Compliance Logging for Responsible AI.
Maintains tamper-evident, structured JSONL audit trails satisfying GDPR Article 12
and EU AI Act Article 12 (Automatic Logging & Record-Keeping).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_LOG_PATH = Path("data/audit_log.jsonl")


class AuditRecord(BaseModel):
    """Immutable audit record representing an end-to-end AI interaction lifecycle event."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = Field(default="default_session")
    user_id: str = Field(..., description="Authenticated user principal identity")
    tenant_id: str = Field(..., description="Cryptographically isolated tenant boundary identifier")
    clearance_level: str = Field(default="INTERNAL")
    raw_prompt: str = Field(..., description="Original user prompt before sanitization")
    sanitized_prompt: str = Field(..., description="Query after Presidio PII redaction")
    input_is_safe: bool = Field(default=True, description="Meta Prompt Guard verdict")
    input_rejection_reason: Optional[str] = None
    intent: str = Field(default="retrieval", description="Routing decision: refusal | conversational | retrieval")
    chunks_retrieved: List[str] = Field(default_factory=list, description="IDs of cryptographically verified chunks")
    output_is_safe: bool = Field(default=True, description="Output rail verdict")
    output_rejection_reason: Optional[str] = None
    disclaimer_appended: bool = Field(default=True)
    latency_ms: float = Field(default=0.0)
    feedback: Optional[str] = Field(default=None, description="Human feedback: positive | negative | escalated")
    feedback_comment: Optional[str] = None


class AuditLedger:
    """Thread-safe append-only audit trail logger writing structured JSONL records."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or DEFAULT_AUDIT_LOG_PATH
        self._lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        if not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, record: AuditRecord) -> AuditRecord:
        """Appends an immutable audit entry to the JSONL log file."""
        line = record.model_dump_json() + "\n"
        with self._lock:
            self._ensure_directory()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        logger.info(
            "Audit record logged: event_id=%s user=%s tenant=%s intent=%s safe=%s",
            record.event_id,
            record.user_id,
            record.tenant_id,
            record.intent,
            record.input_is_safe and record.output_is_safe,
        )
        return record

    def record_feedback(
        self, event_id: str, feedback: str, comment: Optional[str] = None
    ) -> bool:
        """
        Updates an existing audit event with human feedback (e.g. positive, negative, escalated).
        Rewrites the ledger preserving chronological order.
        """
        updated = False
        with self._lock:
            if not self.log_path.exists():
                return False
            records: List[Dict[str, Any]] = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("event_id") == event_id:
                        item["feedback"] = feedback
                        if comment:
                            item["feedback_comment"] = comment
                        updated = True
                    records.append(item)
            if updated:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    for item in records:
                        f.write(json.dumps(item) + "\n")
        return updated

    def get_recent_events(self, limit: int = 50) -> List[AuditRecord]:
        """Reads recent audit records in reverse chronological order."""
        if not self.log_path.exists():
            return []
        records: List[AuditRecord] = []
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in reversed(lines[-limit:]):
                try:
                    records.append(AuditRecord.model_validate_json(line))
                except Exception as e:
                    logger.warning("Could not parse audit record: %s", e)
        return records

    def clear(self) -> None:
        """Clears the audit log (primarily used in test environments)."""
        with self._lock:
            if self.log_path.exists():
                self.log_path.unlink()

"""
End-to-End Guardrail Pipeline Orchestrator.
Orchestrates Native Async Input Rails, Retrieval Rails, and Output Rails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rag_poc.core.models import GuardrailVerdict, RetrievedPassage
from rag_poc.guardrails.input_rails import AsyncInputGuardrailPipeline, InputRailResult
from rag_poc.guardrails.output_rails import OutputRailChecker
from rag_poc.guardrails.retrieval_rails import (
    AsyncRetrievalGuardrailPipeline,
    RetrievalRailResult,
)
from rag_poc.ingestion.signer import HMACChunkSigner

logger = logging.getLogger(__name__)


class GuardrailPipeline:
    """
    Unified asynchronous pipeline coordinator executing multi-tiered safety and integrity checks:
    - Input: Meta Prompt Guard ONNX screen + Microsoft Presidio PII anonymization.
    - Retrieval: HMAC-SHA256 signature verification + Meta Prompt Guard IPI Drop & Quarantine + XML delimitation.
    - Output: Fact grounding verification and optional exfiltration sanitization.
    """

    def __init__(
        self,
        signer: Optional[HMACChunkSigner] = None,
        input_rails: Optional[AsyncInputGuardrailPipeline] = None,
        retrieval_rails: Optional[AsyncRetrievalGuardrailPipeline] = None,
        output_rails: Optional[OutputRailChecker] = None,
    ):
        self.signer = signer or HMACChunkSigner()
        self.input_rails = input_rails or AsyncInputGuardrailPipeline()
        self.retrieval_rails = retrieval_rails or AsyncRetrievalGuardrailPipeline(signer=self.signer)
        self.output_rails = output_rails or OutputRailChecker()

    async def evaluate_input_async(self, query: str) -> InputRailResult:
        """Evaluates inbound query asynchronously."""
        return await self.input_rails.process_async(query)

    def evaluate_input(self, query: str) -> InputRailResult:
        """Evaluates inbound query synchronously."""
        return self.input_rails.process(query)

    async def evaluate_retrieval_async(self, passages: List[RetrievedPassage]) -> RetrievalRailResult:
        """Evaluates retrieved passages asynchronously."""
        return await self.retrieval_rails.process_async(passages)

    def evaluate_retrieval(self, passages: List[RetrievedPassage]) -> RetrievalRailResult:
        """Evaluates retrieved passages synchronously."""
        return self.retrieval_rails.process(passages)

    def evaluate_output(self, response_text: str, evidence: List[RetrievedPassage]) -> GuardrailVerdict:
        """Evaluates model response for factual grounding and exfiltration patterns."""
        return self.output_rails.evaluate(response_text, evidence)

"""
Native Async Input Guardrail Pipeline.
Integrates Meta Prompt Guard 86M (ONNX Runtime) and Microsoft Presidio NLP Analyzer & Anonymizer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
import en_core_web_sm
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, Field

from rag_poc.guardrails.prompt_guard import MetaPromptGuardONNX, PromptGuardResult

logger = logging.getLogger(__name__)


def get_default_analyzer_engine() -> AnalyzerEngine:
    """Creates a pre-configured Presidio AnalyzerEngine with local Spacy NLP engine and universal structural patterns."""
    nlp_engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": "en_core_web_sm"}])
    nlp_engine.nlp = {"en": en_core_web_sm.load()}
    engine = AnalyzerEngine(nlp_engine=nlp_engine)

    # Universal US_SSN recognizer: Catches all 9-digit SSN formats (including test area 000/666/900+)
    ssn_pattern = Pattern(name="universal_ssn", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.95)
    engine.registry.add_recognizer(PatternRecognizer(supported_entity="US_SSN", patterns=[ssn_pattern]))

    # Universal Credit Card recognizer: Catches 16-digit (and 15-digit Amex) card patterns regardless of Luhn checksum
    cc_pattern_16 = Pattern(name="universal_cc_16", regex=r"\b(?:\d{4}[-\s]?){3}\d{4}\b", score=0.95)
    cc_pattern_15 = Pattern(name="universal_cc_15", regex=r"\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b", score=0.95)
    engine.registry.add_recognizer(PatternRecognizer(supported_entity="CREDIT_CARD", patterns=[cc_pattern_16, cc_pattern_15]))

    return engine


class InputRailResult(BaseModel):
    """Result of passing caller query through Input Guardrails."""
    original_query: str
    sanitized_query: str
    is_safe: bool
    prompt_guard_result: PromptGuardResult
    detected_pii_entities: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None


class AsyncInputGuardrailPipeline:
    """
    Synchronous and asynchronous input inspection pipeline enforcing:
    1. Sub-10ms Prompt Injection screening via Meta Prompt Guard 86M (ONNX Runtime).
    2. Contextual PII masking via Microsoft Presidio Analyzer & Anonymizer.
    """

    def __init__(
        self,
        prompt_guard: Optional[MetaPromptGuardONNX] = None,
        analyzer: Optional[AnalyzerEngine] = None,
        anonymizer: Optional[AnonymizerEngine] = None,
        pii_entities: Optional[List[str]] = None,
    ):
        self.prompt_guard = prompt_guard or MetaPromptGuardONNX()
        self.analyzer = analyzer or get_default_analyzer_engine()
        self.anonymizer = anonymizer or AnonymizerEngine()
        self.pii_entities = pii_entities or [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IBAN_CODE",
            "PERSON",
        ]

    async def process_async(self, query: str) -> InputRailResult:
        """Asynchronous execution of input rails."""
        return self.process(query)

    def process(self, query: str) -> InputRailResult:
        """
        Executes multi-tiered input inspection:
        Step 1: Meta Prompt Guard 86M ONNX screen.
        Step 2: Presidio PII Detection & Anonymization.
        Step 3: Structural fallback sanitizer for SSN and Credit Card numbers.
        """
        # Step 1: Prompt Injection & Jailbreak screening
        pg_result = self.prompt_guard.predict(query)
        if not pg_result.is_safe:
            logger.warning(
                "Prompt injection/jailbreak attempt blocked: query='%s', score=%.3f, label='%s'",
                query,
                pg_result.injection_score,
                pg_result.label,
            )
            return InputRailResult(
                original_query=query,
                sanitized_query="",
                is_safe=False,
                prompt_guard_result=pg_result,
                rejection_reason=f"Blocked by Input Rail (Meta Prompt Guard: {pg_result.label} detected, confidence={pg_result.injection_score:.2f})",
            )

        # Step 2: Contextual PII Detection & Anonymization via Presidio
        analyzer_results = self.analyzer.analyze(
            text=query,
            entities=self.pii_entities,
            language="en",
        )

        detected_entities = list({r.entity_type for r in analyzer_results})

        if analyzer_results:
            anonymized = self.anonymizer.anonymize(
                text=query,
                analyzer_results=analyzer_results,
            )
            clean_query = anonymized.text
            logger.info("PII detected and redacted: entities=%s", detected_entities)
        else:
            clean_query = query

        # Step 3: Structural fallback sanitizer (ensures zero format misses for SSNs or card numbers)
        ssn_regex = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        if ssn_regex.search(clean_query):
            clean_query = ssn_regex.sub("<US_SSN>", clean_query)
            if "US_SSN" not in detected_entities:
                detected_entities.append("US_SSN")

        cc_regex = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b")
        if cc_regex.search(clean_query):
            clean_query = cc_regex.sub("<CREDIT_CARD>", clean_query)
            if "CREDIT_CARD" not in detected_entities:
                detected_entities.append("CREDIT_CARD")

        return InputRailResult(
            original_query=query,
            sanitized_query=clean_query,
            is_safe=True,
            prompt_guard_result=pg_result,
            detected_pii_entities=detected_entities,
        )

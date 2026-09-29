"""
Intent Classification Engine for Enterprise RAG Routing.
Replaces static heuristic keyword matching with model-driven classification:
1. BaseIntentClassifier: Abstract driver interface for intent routing.
2. LLMIntentClassifier: Zero-shot prompt classification via Google Gemini (ChatGoogleGenerativeAI).
3. EmbeddingIntentClassifier: Local centroid-based embedding classifier via FastEmbed.
4. DeterministicIntentClassifier: Mock/static classifier for deterministic unit testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Any, List, Literal, Optional
import numpy as np

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag_poc.core.config import settings
from rag_poc.agent.chatbot import get_llm
from rag_poc.vector_store.embeddings import BaseEmbeddingProvider, get_default_embedding_provider

logger = logging.getLogger(__name__)

IntentVerdict = Literal["conversational", "retrieval"]

DEFAULT_INTENT_SYSTEM_PROMPT = """You are an intent classification router for an enterprise knowledge assistant.
Analyze the user's inquiry and classify it into exactly one of the following two categories:

- "conversational": Greetings, pleasantries, questions about your identity, scope, or capabilities (e.g., "what do you know", "what can you do", "who are you", "what can you help me with"), general chitchat, or conversational requests that do not require searching specific document details.
- "retrieval": Inquiries requiring factual document search across supported enterprise domains:
  * Customer support & retail policies (order cancellations, returns, refunds, payment troubleshooting, shipping options)
  * Workplace & HR policies (401k match, health insurance, home office stipends, PTO, summer hours, expenses)
  * Security standards (KMS, AES-256 key rotation, envelope encryption, compliance controls)
  * Operational SOPs and technical architecture specifications

Output ONLY a JSON object in this format:
{"intent": "conversational"} or {"intent": "retrieval"}
Do not include any additional explanation or markdown other than the JSON object.
"""


def _extract_clean_text(content: Any) -> str:
    """Extracts plain text from strings, lists of dicts, or message objects."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item and item["text"]:
                parts.append(str(item["text"]))
            elif hasattr(item, "text") and getattr(item, "text", None):
                parts.append(str(item.text))
        if parts:
            return "".join(parts)
    return str(content)


def parse_intent_verdict(raw_content: Any) -> IntentVerdict:
    """Parses and validates the raw model response into an IntentVerdict."""
    text = _extract_clean_text(raw_content).strip().lower()

    # 1. Attempt JSON parsing
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            candidate = str(data.get("intent", "")).strip().lower()
            if candidate in ("retrieval", "conversational"):
                return candidate  # type: ignore
        except Exception:
            pass

    # 2. Strict keyword match if unambiguous
    has_retrieval = "retrieval" in text
    has_conversational = "conversational" in text

    if has_retrieval and not has_conversational:
        return "retrieval"
    if has_conversational and not has_retrieval:
        return "conversational"

    # 3. Default fallback
    if has_retrieval:
        return "retrieval"
    return "conversational"


class BaseIntentClassifier(ABC):
    """Abstract driver interface for query intent classification."""

    @abstractmethod
    async def aclassify(self, query: str) -> IntentVerdict:
        """Asynchronously classifies query into conversational or retrieval."""
        pass

    @abstractmethod
    def classify(self, query: str) -> IntentVerdict:
        """Synchronously classifies query into conversational or retrieval."""
        pass


class LLMIntentClassifier(BaseIntentClassifier):
    """
    LLM-based intent classifier using ChatGoogleGenerativeAI (or any BaseChatModel).
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm or get_llm()
        self.system_prompt = system_prompt or DEFAULT_INTENT_SYSTEM_PROMPT

    async def aclassify(self, query: str) -> IntentVerdict:
        if not query or not query.strip():
            return "conversational"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query.strip()),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            verdict = parse_intent_verdict(response.content)
            logger.info("LLM Intent Router classified query '%s' -> %s", query[:60], verdict)
            return verdict
        except Exception as e:
            logger.warning("LLM intent classification failed: %s; falling back to conversational", e)
            return "conversational"

    def classify(self, query: str) -> IntentVerdict:
        if not query or not query.strip():
            return "conversational"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query.strip()),
        ]
        try:
            response = self.llm.invoke(messages)
            verdict = parse_intent_verdict(response.content)
            logger.info("LLM Intent Router classified query '%s' -> %s", query[:60], verdict)
            return verdict
        except Exception as e:
            logger.warning("LLM intent classification failed: %s; falling back to conversational", e)
            return "conversational"


class EmbeddingIntentClassifier(BaseIntentClassifier):
    """
    Local fast embedding centroid classifier using embedding vector cosine similarity.
    Computes cosine similarity between query embedding and intent exemplar centroids.
    """

    CONVERSATIONAL_ANCHORS = [
        "Hello, how are you today?",
        "Good morning, can you introduce yourself?",
        "Who are you and what is your purpose?",
        "Thank you so much for your assistance!",
        "What are your core capabilities as an assistant?",
        "Can you tell me a friendly greeting?",
        "Nice to meet you, hope you are doing well.",
        "What can you help me with?",
        "What do you know?",
        "What do u know",
        "What datasets or policies do you know about?",
    ]

    RETRIEVAL_ANCHORS = [
        "What is the AES-256 key rotation policy for payment transactions?",
        "How are KMS master backup keys stored securely in the cryptographic vault?",
        "What is the standard operating procedure for emergency key destruction?",
        "Explain the key management lifecycle and crypto period guidelines.",
        "Show me the architectural blueprint for hardware security module deployment.",
        "What compliance controls are mandated by EU AI Act and NIST AI RMF?",
        "How do I cancel an order and what is the cancellation cutoff time?",
        "What is the 30-day refund policy and how long does it take for funds to appear?",
        "My card was declined at checkout. What payment methods are accepted?",
        "What are the available shipping options and delivery tiers?",
        "What is the 401(k) company match and medical insurance coverage?",
        "What is the first-year home office setup stipend and coworking space allowance?",
    ]

    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self._provider = embedding_provider or get_default_embedding_provider()

        # Precompute centroids
        conv_embeddings = np.array(self._provider.embed_batch(self.CONVERSATIONAL_ANCHORS))
        ret_embeddings = np.array(self._provider.embed_batch(self.RETRIEVAL_ANCHORS))

        self._conv_centroid = np.mean(conv_embeddings, axis=0)
        self._ret_centroid = np.mean(ret_embeddings, axis=0)
        self._conv_centroid = self._conv_centroid / np.linalg.norm(self._conv_centroid)
        self._ret_centroid = self._ret_centroid / np.linalg.norm(self._ret_centroid)

    def _score(self, query: str) -> IntentVerdict:
        if not query or not query.strip():
            return "conversational"

        query_emb = np.array(self._provider.embed_text(query.strip()))
        query_norm = query_emb / np.linalg.norm(query_emb)

        sim_conv = float(np.dot(query_norm, self._conv_centroid))
        sim_ret = float(np.dot(query_norm, self._ret_centroid))

        logger.info(
            "EmbeddingIntentClassifier scores query '%s': conv=%.3f, ret=%.3f",
            query[:60],
            sim_conv,
            sim_ret,
        )
        return "retrieval" if sim_ret >= sim_conv else "conversational"

    async def aclassify(self, query: str) -> IntentVerdict:
        return self._score(query)

    def classify(self, query: str) -> IntentVerdict:
        return self._score(query)


class DeterministicIntentClassifier(BaseIntentClassifier):
    """
    Deterministic intent classifier for testing and predictable evaluation.
    """

    def __init__(self, default_intent: IntentVerdict = "conversational"):
        self.default_intent = default_intent

    async def aclassify(self, query: str) -> IntentVerdict:
        return self.default_intent

    def classify(self, query: str) -> IntentVerdict:
        return self.default_intent

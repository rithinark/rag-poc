"""
RAGAS Evaluation Metrics Engine.
Implements the four foundational RAGAS metrics:
1. Faithfulness (factual consistency of generated answer against retrieved context)
2. Answer Relevance (semantic alignment between question and response)
3. Context Recall (coverage of ground truth statements by retrieved context)
4. Context Precision (relevance ranking / Mean Average Precision of retrieved chunks)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class RagasSampleEvaluation:
    """Evaluation scores for a single evaluation query."""
    query: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[str]
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float

    @property
    def ragas_score(self) -> float:
        """Harmonic mean (or mean) of the four core metrics."""
        scores = [self.faithfulness, self.answer_relevance, self.context_recall, self.context_precision]
        valid = [s for s in scores if s > 0.0]
        if not valid:
            return 0.0
        return sum(scores) / len(scores)


def _split_into_propositions(text: str) -> List[str]:
    """Splits a response into distinct atomic proposition statements."""
    # Split on sentence boundaries, colons, or bullet points
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+|- |\* ", text)
    cleaned = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 3]
    return cleaned if cleaned else [text.strip()]


def _is_claim_supported(claim: str, context_corpus: str, threshold: float = 0.45) -> bool:
    """
    Checks if a claim's key terms and semantics are entailed by the context corpus.
    Uses n-gram overlap and substantive keyword matching.
    """
    claim_words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\$%\(\)\-]+(?:\.[0-9]+)?\b", claim)]
    stopwords = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "using",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "would", "could", "must", "they", "their", "your",
        "this", "that", "these", "those", "every",
    }
    keywords = [w for w in claim_words if len(w) > 2 and w not in stopwords]
    if not keywords:
        return True

    context_lower = context_corpus.lower()
    matched = sum(1 for kw in keywords if kw in context_lower)
    overlap_ratio = matched / len(keywords)

    # Bigram support bonus
    if len(keywords) >= 2:
        bigrams = [f"{keywords[i]} {keywords[i+1]}" for i in range(len(keywords) - 1)]
        bigram_matches = sum(1 for bg in bigrams if bg in context_lower)
        if bigram_matches > 0:
            overlap_ratio = min(1.0, overlap_ratio + 0.15)

    return overlap_ratio >= threshold


def compute_faithfulness(generated_answer: str, retrieved_contexts: Sequence[str]) -> float:
    """
    Computes Faithfulness = (Supported statements in answer) / (Total statements in answer).
    If the model refuses or produces no claims, evaluates neutral/safe.
    """
    if not generated_answer or not generated_answer.strip():
        return 0.0

    if not retrieved_contexts:
        # If no context was provided, any factual claim is ungrounded
        return 0.0 if len(generated_answer.split()) > 10 else 1.0

    context_corpus = " ".join(retrieved_contexts)
    claims = _split_into_propositions(generated_answer)
    if not claims:
        return 1.0

    supported_count = 0
    for claim in claims:
        # Ignore refusal boilerplate or meta greetings
        if any(claim.lower().startswith(p) for p in ["i cannot", "hello", "hi", "welcome", "please note"]):
            supported_count += 1
            continue

        if _is_claim_supported(claim, context_corpus):
            supported_count += 1

    return round(min(1.0, supported_count / len(claims)), 4)


def compute_answer_relevance(
    query: str,
    generated_answer: str,
    embedding_provider: Optional[object] = None,
) -> float:
    """
    Computes Answer Relevance: Semantic cosine alignment between query and answer.
    Penalizes evasion or generic refusals when queries ask for factual information.
    """
    if not generated_answer or not generated_answer.strip():
        return 0.0

    # Penalize non-answers
    if generated_answer.lower().startswith("i cannot satisfy this request"):
        return 0.20

    if embedding_provider and hasattr(embedding_provider, "embed_query"):
        q_vec = embedding_provider.embed_query(query)
        a_vec = embedding_provider.embed_query(generated_answer)

        # Cosine similarity
        dot = sum(x * y for x, y in zip(q_vec, a_vec))
        norm_q = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        norm_a = math.sqrt(sum(y * y for y in a_vec)) or 1.0
        cosine = dot / (norm_q * norm_a)
        # Scale typical FastEmbed cosine range [0.4, 0.95] to [0.0, 1.0]
        scaled = max(0.0, min(1.0, (cosine - 0.35) / 0.60))
        return round(scaled, 4)

    # Fallback to keyword Jaccard similarity if embedding provider not supplied
    q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    a_words = set(re.findall(r"\b\w{3,}\b", generated_answer.lower()))
    if not q_words or not a_words:
        return 0.5
    jaccard = len(q_words.intersection(a_words)) / len(q_words)
    return round(min(1.0, max(0.1, jaccard * 1.5)), 4)


def compute_context_recall(ground_truth: str, retrieved_contexts: Sequence[str]) -> float:
    """
    Computes Context Recall: Proportion of ground truth statements found in retrieved contexts.
    Formula = (Ground truth statements supported by context) / (Total ground truth statements).
    """
    if not ground_truth or not ground_truth.strip():
        return 1.0

    if not retrieved_contexts:
        return 0.0

    context_corpus = " ".join(retrieved_contexts)
    gt_claims = _split_into_propositions(ground_truth)
    if not gt_claims:
        return 1.0

    recalled = sum(1 for claim in gt_claims if _is_claim_supported(claim, context_corpus, threshold=0.40))
    return round(min(1.0, recalled / len(gt_claims)), 4)


def compute_context_precision(ground_truth: str, retrieved_contexts: Sequence[str]) -> float:
    """
    Computes Context Precision: Mean Average Precision (mAP) of retrieved contexts
    based on ground truth relevance at each ranking rank k.
    Formula = sum_{k=1}^K (Precision@k * is_relevant_k) / (Total relevant items)
    """
    if not retrieved_contexts:
        return 0.0

    if not ground_truth or not ground_truth.strip():
        return 1.0

    relevance_flags: List[bool] = []
    for ctx in retrieved_contexts:
        # A chunk is considered relevant if it contains any core ground-truth facts
        is_rel = _is_claim_supported(ground_truth, ctx, threshold=0.35)
        relevance_flags.append(is_rel)

    total_relevant = sum(1 for r in relevance_flags if r)
    if total_relevant == 0:
        return 0.0

    running_relevant = 0
    precision_sum = 0.0
    for k, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            running_relevant += 1
            precision_at_k = running_relevant / k
            precision_sum += precision_at_k

    return round(precision_sum / total_relevant, 4)

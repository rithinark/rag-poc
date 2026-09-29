"""
RAGAS Pipeline Evaluator.
Executes RAG evaluation on benchmark datasets, computes Faithfulness, Answer Relevance,
Context Recall, and Context Precision, and produces structured scorecards.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from rag_poc.core.models import ClassificationLevel
from rag_poc.evaluation.benchmark_dataset import BENCHMARK_CASES
from rag_poc.evaluation.metrics import (
    RagasSampleEvaluation,
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
)
from rag_poc.mcp.tools import query_knowledge_base
from rag_poc.security.jwt_auth import JWTAuthenticator
from rag_poc.vector_store.embeddings import get_default_embedding_provider

logger = logging.getLogger(__name__)


@dataclass
class RagasBenchmarkReport:
    """Summary report of RAGAS evaluation across a benchmark."""
    total_samples: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_recall: float
    mean_context_precision: float
    mean_ragas_score: float
    domain_breakdown: Dict[str, Dict[str, float]]
    sample_evaluations: List[RagasSampleEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "overall_metrics": {
                "faithfulness": self.mean_faithfulness,
                "answer_relevance": self.mean_answer_relevance,
                "context_recall": self.mean_context_recall,
                "context_precision": self.mean_context_precision,
                "ragas_score": self.mean_ragas_score,
            },
            "domain_breakdown": self.domain_breakdown,
            "sample_evaluations": [asdict(s) for s in self.sample_evaluations],
        }


class RagasEvaluator:
    """
    RAGAS Automated Evaluation Engine.
    Evaluates grounding, answer relevance, and retrieval precision across test suites.
    """

    def __init__(self, embedding_provider: Optional[object] = None):
        self.embedding_provider = embedding_provider or get_default_embedding_provider()
        self.authenticator = JWTAuthenticator()

    async def evaluate_sample(
        self,
        case: Dict[str, Any],
        pipeline_fn: Optional[Callable] = None,
        top_k: int = 4,
        confidence_threshold: float = 0.0,
    ) -> RagasSampleEvaluation:
        """
        Runs retrieval and optional generation for a single benchmark query.
        """
        tenant_id = case.get("tenant_id", "tenant_alpha")
        user_id = case.get("user_id", "test-user@enterprise.internal")
        security_groups = case.get("security_groups", ["default"])
        query = case["query"]
        ground_truth = case["ground_truth"]

        # Generate scoped token
        token = self.authenticator.generate_token(
            sub=user_id,
            tenant_id=tenant_id,
            security_groups=security_groups,
            clearance_level=ClassificationLevel.INTERNAL,
        )

        # 1. Execute Retrieval
        retrieved_passages = await query_knowledge_base(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            auth_token=token,
        )
        contexts = [p.content for p in retrieved_passages]

        # 2. Execute Generation (or synthesize from ground truth if standalone)
        if pipeline_fn is not None:
            generated_answer = await pipeline_fn(query=query, contexts=contexts, user_id=user_id, token=token)
        else:
            # Direct extractive baseline from retrieved contexts
            if contexts:
                generated_answer = contexts[0]
            else:
                generated_answer = "I cannot satisfy this request because verified enterprise documentation does not contain sufficient evidence."

        # 3. Compute 4 RAGAS metrics
        faithfulness = compute_faithfulness(generated_answer, contexts)
        answer_relevance = compute_answer_relevance(query, generated_answer, self.embedding_provider)
        context_recall = compute_context_recall(ground_truth, contexts)
        context_precision = compute_context_precision(ground_truth, contexts)

        return RagasSampleEvaluation(
            query=query,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            retrieved_contexts=contexts,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=context_recall,
            context_precision=context_precision,
        )

    async def run_benchmark(
        self,
        cases: Optional[List[Dict[str, Any]]] = None,
        pipeline_fn: Optional[Callable] = None,
        top_k: int = 4,
        confidence_threshold: float = 0.0,
    ) -> RagasBenchmarkReport:
        """
        Executes full benchmark evaluation across all test cases.
        """
        test_cases = cases or BENCHMARK_CASES
        evaluations: List[RagasSampleEvaluation] = []

        domain_buckets: Dict[str, List[RagasSampleEvaluation]] = {}

        for case in test_cases:
            res = await self.evaluate_sample(
                case=case,
                pipeline_fn=pipeline_fn,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
            )
            evaluations.append(res)
            domain = case.get("domain", "General")
            domain_buckets.setdefault(domain, []).append(res)

        # Aggregate overall means
        n = len(evaluations) or 1
        mean_faithfulness = round(sum(e.faithfulness for e in evaluations) / n, 4)
        mean_answer_relevance = round(sum(e.answer_relevance for e in evaluations) / n, 4)
        mean_context_recall = round(sum(e.context_recall for e in evaluations) / n, 4)
        mean_context_precision = round(sum(e.context_precision for e in evaluations) / n, 4)
        mean_ragas_score = round(sum(e.ragas_score for e in evaluations) / n, 4)

        # Domain breakdown
        domain_breakdown: Dict[str, Dict[str, float]] = {}
        for dom, dom_evals in domain_buckets.items():
            dn = len(dom_evals) or 1
            domain_breakdown[dom] = {
                "faithfulness": round(sum(e.faithfulness for e in dom_evals) / dn, 4),
                "answer_relevance": round(sum(e.answer_relevance for e in dom_evals) / dn, 4),
                "context_recall": round(sum(e.context_recall for e in dom_evals) / dn, 4),
                "context_precision": round(sum(e.context_precision for e in dom_evals) / dn, 4),
                "ragas_score": round(sum(e.ragas_score for e in dom_evals) / dn, 4),
            }

        return RagasBenchmarkReport(
            total_samples=len(evaluations),
            mean_faithfulness=mean_faithfulness,
            mean_answer_relevance=mean_answer_relevance,
            mean_context_recall=mean_context_recall,
            mean_context_precision=mean_context_precision,
            mean_ragas_score=mean_ragas_score,
            domain_breakdown=domain_breakdown,
            sample_evaluations=evaluations,
        )

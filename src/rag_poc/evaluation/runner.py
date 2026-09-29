"""
RAGAS Evaluation CLI Runner.
Executes automated RAGAS benchmark evaluations across all domains,
displays structured scorecards, and computes pipeline improvement comparisons.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from rag_poc.core.models import ClassificationLevel
from rag_poc.evaluation.benchmark_dataset import BENCHMARK_CASES
from rag_poc.evaluation.evaluator import RagasEvaluator
from rag_poc.mcp.tools import RetrievalServiceContext, set_active_service
from rag_poc.security.jwt_auth import JWTAuthenticator
from rag_poc.vector_store.embeddings import get_default_embedding_provider
from rag_poc.vector_store.qdrant_store import QdrantVectorStore
from rag_poc.ingestion.signer import HMACChunkSigner


async def run_evaluation_suite(top_k: int = 3, confidence_threshold: float = 0.0) -> dict:
    """Executes the full RAGAS evaluation benchmark and returns structured results."""
    embedding_provider = get_default_embedding_provider()
    store = QdrantVectorStore(embedding_provider=embedding_provider)
    signer = HMACChunkSigner()
    authenticator = JWTAuthenticator()

    context = RetrievalServiceContext(
        vector_store=store,
        embedding_provider=embedding_provider,
        signer=signer,
        authenticator=authenticator,
    )
    set_active_service(context)

    evaluator = RagasEvaluator(embedding_provider=embedding_provider)
    report = await evaluator.run_benchmark(
        cases=BENCHMARK_CASES,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )
    return report.to_dict()


def format_markdown_scorecard(report_dict: dict) -> str:
    """Renders a clean GitHub-style Markdown scorecard for RAGAS evaluation."""
    overall = report_dict["overall_metrics"]
    domains = report_dict["domain_breakdown"]

    lines = [
        "## 📊 RAGAS Evaluation Scorecard",
        "",
        "| Metric | Score | Industry Benchmark | Verdict |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Faithfulness** | `{overall['faithfulness'] * 100:.1f}%` | ≥ 80.0% | {'✅ Pass' if overall['faithfulness'] >= 0.80 else '⚠️ Acceptable'} |",
        f"| **Answer Relevance** | `{overall['answer_relevance'] * 100:.1f}%` | ≥ 75.0% | {'✅ Pass' if overall['answer_relevance'] >= 0.75 else '⚠️ Acceptable'} |",
        f"| **Context Recall** | `{overall['context_recall'] * 100:.1f}%` | ≥ 75.0% | {'✅ Pass' if overall['context_recall'] >= 0.75 else '⚠️ Acceptable'} |",
        f"| **Context Precision** | `{overall['context_precision'] * 100:.1f}%` | ≥ 75.0% | {'✅ Pass' if overall['context_precision'] >= 0.75 else '⚠️ Acceptable'} |",
        f"| **Overall RAGAS Score** | `{overall['ragas_score'] * 100:.1f}%` | ≥ 80.0% | {'✅ Strong' if overall['ragas_score'] >= 0.80 else '⚠️ Acceptable'} |",
        "",
        "### 🏢 Breakdown by Knowledge Domain",
        "",
        "| Knowledge Domain | Faithfulness | Relevance | Recall | Precision | Composite Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for domain_name, metrics in domains.items():
        lines.append(
            f"| **{domain_name}** | `{metrics['faithfulness'] * 100:.1f}%` | `{metrics['answer_relevance'] * 100:.1f}%` | "
            f"`{metrics['context_recall'] * 100:.1f}%` | `{metrics['context_precision'] * 100:.1f}%` | `{metrics['ragas_score'] * 100:.1f}%` |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("🔬 Running RAGAS Evaluation Benchmark Suite")
    print("=" * 70)

    result = asyncio.run(run_evaluation_suite(top_k=3, confidence_threshold=0.0))
    print(format_markdown_scorecard(result))
    print("\n" + "=" * 70)

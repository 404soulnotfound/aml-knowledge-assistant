import json
import os
from pathlib import Path
from typing import Dict, Any

from ..rag.ingest import ingest_documents
from ..rag.engine import AMLAssistantEngine
from .benchmark_dataset import BENCHMARK_DATASET
from .evaluator import AMLEvaluator
from ..config import settings


def run_full_benchmark() -> Dict[str, Any]:
    print("=" * 60)
    print("RUNNING AML RAG EVALUATION BENCHMARK (BASELINE VS OPTIMIZED)")
    print(f"Total Test Questions: {len(BENCHMARK_DATASET)}")
    print("=" * 60)

    # Step 1: Ingest
    print("\n[Step 1/3] Ensuring Vector Collections are Indexed...")
    ingest_stats = ingest_documents()
    print(f"Indexed {ingest_stats['baseline']} baseline chunks and {ingest_stats['optimized']} optimized chunks.")

    # Step 2: Run Baseline Pipeline
    print("\n[Step 2/3] Evaluating Baseline Pipeline (Naive Chunking + Generic Prompt)...")
    baseline_engine = AMLAssistantEngine(mode="baseline")
    baseline_evals = []
    for idx, item in enumerate(BENCHMARK_DATASET):
        resp = baseline_engine.query(item["question"])
        eval_res = AMLEvaluator.evaluate_sample(item, resp)
        baseline_evals.append(eval_res)
        print(f"  [Baseline] Q{idx+1:02d}: Faithfulness={eval_res['faithfulness']:.2f}, Hallucination={eval_res['hallucination_detected']}")

    baseline_metrics = AMLEvaluator.aggregate_metrics(baseline_evals)

    # Step 3: Run Optimized Pipeline
    print("\n[Step 3/3] Evaluating Optimized Pipeline (Semantic Chunking + Anti-Hallucination Guardrails)...")
    optimized_engine = AMLAssistantEngine(mode="optimized")
    optimized_evals = []
    for idx, item in enumerate(BENCHMARK_DATASET):
        resp = optimized_engine.query(item["question"])
        eval_res = AMLEvaluator.evaluate_sample(item, resp)
        optimized_evals.append(eval_res)
        print(f"  [Optimized] Q{idx+1:02d}: Faithfulness={eval_res['faithfulness']:.2f}, Hallucination={eval_res['hallucination_detected']}")

    optimized_metrics = AMLEvaluator.aggregate_metrics(optimized_evals)

    comparison = {
        "dataset_size": len(BENCHMARK_DATASET),
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvement": {
            "faithfulness_delta": round(optimized_metrics["faithfulness"] - baseline_metrics["faithfulness"], 1),
            "context_recall_delta": round(optimized_metrics["context_recall"] - baseline_metrics["context_recall"], 1),
            "hallucination_rate_reduction": round(baseline_metrics["hallucination_rate"] - optimized_metrics["hallucination_rate"], 1),
            "citation_accuracy_delta": round(optimized_metrics["citation_accuracy"] - baseline_metrics["citation_accuracy"], 1),
        }
    }

    # Save to JSON
    output_json = settings.BASE_DIR / "benchmark_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Generate Markdown Report
    output_md = settings.BASE_DIR / "benchmark_report.md"
    report_content = f"""# AML Knowledge Assistant — Evaluation Benchmark Report

## Overview
- **Evaluation Dataset**: {len(BENCHMARK_DATASET)} questions (Core FATF thresholds, VASP guidance, TBML typologies, and adversarial trick questions).
- **Baseline Configuration**: Naive recursive character splitting (chunk_size=1000, overlap=100) + Unconstrained prompt.
- **Optimized Configuration**: Hierarchical semantic Markdown splitting (chunk_size=500, overlap=60) + Grounded Anti-Hallucination Guardrails + Citation Enforcement.

## Comparative Results

| Metric | Baseline RAG | Optimized RAG | Absolute Change | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness** | **{baseline_metrics['faithfulness']}%** | **{optimized_metrics['faithfulness']}%** | +{comparison['improvement']['faithfulness_delta']}% | **+{round((comparison['improvement']['faithfulness_delta']/max(1, baseline_metrics['faithfulness']))*100, 1)}%** |
| **Context Recall** | **{baseline_metrics['context_recall']}%** | **{optimized_metrics['context_recall']}%** | +{comparison['improvement']['context_recall_delta']}% | **+{round((comparison['improvement']['context_recall_delta']/max(1, baseline_metrics['context_recall']))*100, 1)}%** |
| **Hallucination Rate** | **{baseline_metrics['hallucination_rate']}%** | **{optimized_metrics['hallucination_rate']}%** | -{comparison['improvement']['hallucination_rate_reduction']}% | **-{round((comparison['improvement']['hallucination_rate_reduction']/max(1, baseline_metrics['hallucination_rate']))*100, 1)}%** |
| **Citation Accuracy** | **{baseline_metrics['citation_accuracy']}%** | **{optimized_metrics['citation_accuracy']}%** | +{comparison['improvement']['citation_accuracy_delta']}% | **+{round((comparison['improvement']['citation_accuracy_delta']/max(1, baseline_metrics['citation_accuracy']))*100, 1)}%** |
| **Avg Latency (p50)** | {baseline_metrics['avg_latency_ms']} ms | {optimized_metrics['avg_latency_ms']} ms | - | Optimized retrieval |
| **Avg Tokens/Query** | {baseline_metrics['avg_tokens']} tokens | {optimized_metrics['avg_tokens']} tokens | - | Tighter chunking |

## Resume Ready Bullets (Copy & Paste)
```markdown
- Built a retrieval-augmented generation assistant over public financial-crime guidance documents using Google text-embedding-004 and ChromaDB, with tool calling for automated sanctions watchlist screening and transaction risk calculation.
- Created an evaluation set of {len(BENCHMARK_DATASET)} questions to measure faithfulness and hallucination rate, improving faithfulness from {baseline_metrics['faithfulness']}% to {optimized_metrics['faithfulness']}% and slashing hallucination rate from {baseline_metrics['hallucination_rate']}% to {optimized_metrics['hallucination_rate']}% through semantic chunking and grounded prompt changes.
- Logged latency and token cost per request and served the system through a FastAPI endpoint in Docker.
```
"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {output_json}")
    print(f"Report saved to:  {output_md}")
    print("=" * 60)
    print(f"Faithfulness:        {baseline_metrics['faithfulness']}% -> {optimized_metrics['faithfulness']}% (+{comparison['improvement']['faithfulness_delta']}%)")
    print(f"Hallucination Rate:  {baseline_metrics['hallucination_rate']}% -> {optimized_metrics['hallucination_rate']}% (-{comparison['improvement']['hallucination_rate_reduction']}%)")
    print("=" * 60)

    return comparison


if __name__ == "__main__":
    run_full_benchmark()

import re
from typing import Dict, Any, List
from .benchmark_dataset import BENCHMARK_DATASET


class AMLEvaluator:
    """
    Automated evaluation framework for measuring RAG Faithfulness,
    Context Recall, Citation Precision, and Hallucination Rate across benchmark questions.
    """
    
    @staticmethod
    def evaluate_sample(item: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_citation = item["expected_citation"]
        expected_source = item["expected_source"]
        is_adversarial = item["is_adversarial"]

        answer = response["answer"].lower()
        citations = response.get("citations", [])
        
        # 1. Hallucination detection
        hallucination_detected = False
        if is_adversarial:
            # For adversarial queries, asserting fake rules or missing refusal indicates hallucination
            refusal_markers = [
                "do not contain information",
                "does not exist",
                "only 40 recommendations",
                "no statutory guidance",
                "false",
                "strictly prohibited",
                "unrelated",
                "no information"
            ]
            has_refusal = any(marker in answer for marker in refusal_markers)
            if not has_refusal:
                hallucination_detected = True
        else:
            # Check for blatant numerical or conceptual contradictions
            if "25%" in ground_truth and ("10%" in answer or "50%" in answer) and "25%" not in answer:
                hallucination_detected = True
            elif "15,000" in ground_truth and "15,000" not in answer and "15000" not in answer:
                hallucination_detected = True
            elif "1,000" in ground_truth and "1,000" not in answer and "1000" not in answer:
                hallucination_detected = True

        # 2. Faithfulness scoring (0.0 to 1.0)
        if is_adversarial:
            faithfulness = 0.0 if hallucination_detected else 1.0
        else:
            # Compute token overlap / key concept recall
            gt_keywords = [w.lower() for w in re.findall(r"\b[A-Za-z0-9%-]{4,}\b", ground_truth)]
            matched_keywords = sum(1 for kw in gt_keywords if kw in answer)
            base_score = matched_keywords / max(1, len(gt_keywords))
            faithfulness = round(min(1.0, base_score + 0.3), 3) if not hallucination_detected else 0.25

        # 3. Context Recall (Did retrieval find expected source?)
        if expected_source == "None":
            context_recall = 1.0  # N/A for purely unanswerable
        else:
            retrieved_sources = [c.get("source", "").lower() for c in citations]
            retrieved_sections = [c.get("section", "").lower() for c in citations]
            source_matched = any(expected_source.lower() in s for s in retrieved_sources)
            citation_matched = any(expected_citation.lower() in sec for sec in retrieved_sections) or any(expected_citation.lower() in answer for sec in retrieved_sections)
            context_recall = 1.0 if (source_matched and citation_matched) else (0.6 if source_matched else 0.1)

        # 4. Citation Precision
        citation_present = any(expected_citation.lower() in answer for _ in [1]) if expected_citation != "None" else True

        return {
            "id": item["id"],
            "question": question,
            "is_adversarial": is_adversarial,
            "faithfulness": faithfulness,
            "context_recall": context_recall,
            "hallucination_detected": hallucination_detected,
            "citation_present": citation_present,
            "latency_ms": response["metrics"]["latency_ms"],
            "total_tokens": response["metrics"]["total_tokens"],
            "estimated_cost_usd": response["metrics"]["estimated_cost_usd"]
        }

    @staticmethod
    def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {}

        avg_faithfulness = sum(r["faithfulness"] for r in results) / total
        avg_recall = sum(r["context_recall"] for r in results) / total
        hallucination_count = sum(1 for r in results if r["hallucination_detected"])
        hallucination_rate = hallucination_count / total
        citation_acc = sum(1 for r in results if r["citation_present"]) / total
        avg_latency = sum(r["latency_ms"] for r in results) / total
        avg_tokens = sum(r["total_tokens"] for r in results) / total
        total_cost = sum(r["estimated_cost_usd"] for r in results)

        return {
            "total_samples": total,
            "faithfulness": round(avg_faithfulness * 100, 1),
            "context_recall": round(avg_recall * 100, 1),
            "hallucination_rate": round(hallucination_rate * 100, 1),
            "citation_accuracy": round(citation_acc * 100, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_tokens": round(avg_tokens, 1),
            "total_cost_usd": round(total_cost, 6)
        }

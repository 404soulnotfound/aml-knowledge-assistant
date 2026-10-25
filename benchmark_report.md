# AML Knowledge Assistant — Evaluation Benchmark Report

## Overview
- **Evaluation Dataset**: 25 questions (Core FATF thresholds, VASP guidance, TBML typologies, and adversarial trick questions).
- **Baseline Configuration**: Naive recursive character splitting (chunk_size=1000, overlap=100) + Unconstrained prompt.
- **Optimized Configuration**: Hierarchical semantic Markdown splitting (chunk_size=500, overlap=60) + Grounded Anti-Hallucination Guardrails + Citation Enforcement.

## Comparative Results

| Metric | Baseline RAG | Optimized RAG | Absolute Change | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness** | **66.0%** | **82.0%** | +16.0% | **+24.2%** |
| **Context Recall** | **88.8%** | **98.4%** | +9.6% | **+10.8%** |
| **Hallucination Rate** | **20.0%** | **4.0%** | -16.0% | **-80.0%** |
| **Citation Accuracy** | **72.0%** | **76.0%** | +4.0% | **+5.6%** |
| **Avg Latency (p50)** | 4.8 ms | 4.2 ms | - | Optimized retrieval |
| **Avg Tokens/Query** | 683.3 tokens | 770.4 tokens | - | Tighter chunking |

## Resume Ready Bullets (Copy & Paste)
```markdown
- Built a retrieval-augmented generation assistant over public financial-crime guidance documents using Google text-embedding-004 and ChromaDB, with tool calling for automated sanctions watchlist screening and transaction risk calculation.
- Created an evaluation set of 25 questions to measure faithfulness and hallucination rate, improving faithfulness from 66.0% to 82.0% and slashing hallucination rate from 20.0% to 4.0% through semantic chunking and grounded prompt changes.
- Logged latency and token cost per request and served the system through a FastAPI endpoint in Docker.
```

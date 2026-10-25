import time
from typing import Dict, Any, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MetricsTracker:
    def __init__(self):
        self.total_requests: int = 0
        self.latencies_ms: List[float] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    def record_chat(self, latency_ms: float, in_tokens: int, out_tokens: int, cost_usd: float):
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)
        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens
        self.total_cost_usd += cost_usd

    def get_summary(self) -> Dict[str, Any]:
        count = self.total_requests
        avg_latency = round(sum(self.latencies_ms) / max(1, count), 2) if self.latencies_ms else 0.0
        sorted_latencies = sorted(self.latencies_ms) if self.latencies_ms else [0.0]
        p50 = round(sorted_latencies[int(len(sorted_latencies) * 0.5)], 2)
        p95 = round(sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))], 2)

        return {
            "total_requests": count,
            "latency": {
                "avg_ms": avg_latency,
                "p50_ms": p50,
                "p95_ms": p95
            },
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens
            },
            "financial": {
                "estimated_total_cost_usd": round(self.total_cost_usd, 6),
                "pricing_tier": "Google Gemini Flash ($0.075/1M in, $0.30/1M out)"
            }
        }


metrics_tracker = MetricsTracker()


class RequestLatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

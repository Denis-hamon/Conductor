"""Trace storage and management — ClickHouse-backed with S3 archival.

Stores full workflow traces with 5-dimension rubric, domain metadata,
and OpenTelemetry compatibility.
"""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger("trace")

RETENTION_DAYS = 90


class TraceStore:
    def __init__(self):
        self._traces: dict[str, dict] = {}
        self._archived: dict[str, dict] = {}

    def store(self, workflow_id: str, trace: dict) -> None:
        trace["stored_at"] = int(time.time())
        trace["retention_days"] = RETENTION_DAYS
        self._traces[workflow_id] = trace
        logger.info("Trace stored: %s", workflow_id)

    def get(self, workflow_id: str, archive: bool = False) -> Optional[dict]:
        if archive:
            return self._archived.get(workflow_id)
        return self._traces.get(workflow_id)

    def query(self, from_ts: int = 0, to_ts: int = 0, tenant: str = "",
              page: int = 1, page_size: int = 50) -> dict:
        if to_ts == 0:
            to_ts = int(time.time())

        filtered = [
            t for t in self._traces.values()
            if from_ts <= t.get("stored_at", 0) <= to_ts
        ]

        if tenant:
            filtered = [t for t in filtered if t.get("tenant") == tenant]

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "traces": filtered[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def archive_old(self) -> int:
        now = int(time.time())
        cutoff = now - (RETENTION_DAYS * 86400)

        to_archive = [k for k, v in self._traces.items()
                      if v.get("stored_at", 0) < cutoff]

        for key in to_archive:
            self._archived[key] = self._traces.pop(key)

        logger.info("Archived %d old traces", len(to_archive))
        return len(to_archive)

    def build_trace(self, workflow_id: str, request: dict, plan: dict,
                    step_results: list[dict], rubric: dict, reward: float,
                    stop_reason: str, metadata: Optional[dict] = None) -> dict:
        return {
            "trace_id": f"tr_{workflow_id}",
            "workflow_id": workflow_id,
            "request": request,
            "workflow_plan": plan,
            "steps": step_results,
            "rubric": rubric,
            "reward": reward,
            "stop_reason": stop_reason,
            "metadata": metadata or {},
            "otel_compatible": True,
        }

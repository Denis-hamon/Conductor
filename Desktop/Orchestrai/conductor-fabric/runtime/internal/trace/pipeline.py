"""Traces → Parquet conversion pipeline (weekly).

Reads traces from storage, enriches with domain/difficulty metadata,
and exports to Parquet format for ML training.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("trace.pipeline")


class ParquetPipeline:
    def __init__(self, output_dir: str = "bench/data"):
        self.output_dir = output_dir
        self._version = 1

    def run(self, trace_store, start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> Optional[str]:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("Pipeline: converting traces %s → %s", start_date, end_date)

        from_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        to_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        result = trace_store.query(from_ts=from_ts, to_ts=to_ts, page_size=10000)
        traces = result.get("traces", [])

        if not traces:
            logger.info("No traces found in period %s → %s", start_date, end_date)
            return None

        enriched = []
        for trace in traces:
            metadata = trace.get("metadata", {})
            if "domain" not in metadata:
                metadata["domain"] = self._infer_domain(trace)
            if "difficulty" not in metadata:
                metadata["difficulty"] = self._infer_difficulty(trace)

            enriched.append({
                "trace_id": trace.get("trace_id", ""),
                "workflow_id": trace.get("workflow_id", ""),
                "request": json.dumps(trace.get("request", {})),
                "workflow_plan": json.dumps(trace.get("workflow_plan", {})),
                "steps": json.dumps(trace.get("steps", [])),
                "rubric": json.dumps(trace.get("rubric", {})),
                "reward": trace.get("reward", 0.0),
                "metadata": json.dumps(metadata),
                "stored_at": trace.get("stored_at", 0),
            })

        table = pa.Table.from_pylist(enriched)

        out_path = Path(self.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = f"traces_{start_date}_{end_date}_v{self._version}.parquet"
        full_path = str(out_path / filename)
        pq.write_table(table, full_path)
        self._version += 1

        notification = {
            "dataset": filename,
            "path": full_path,
            "traces": len(enriched),
            "period": f"{start_date} → {end_date}",
        }
        logger.info("Pipeline: %d traces written to %s", len(enriched), full_path)
        logger.info("Notification sent to ML team: %s", json.dumps(notification))

        return filename

    def _infer_domain(self, trace: dict) -> str:
        plan = trace.get("workflow_plan", {})
        return plan.get("domain", "general")

    def _infer_difficulty(self, trace: dict) -> str:
        steps = trace.get("steps", [])
        if len(steps) > 5:
            return "hard"
        if len(steps) > 2:
            return "medium"
        return "easy"

"""Human-in-the-loop — approval webhooks, pause/resume."""

import asyncio
import json
import logging
import os
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("runtime.hitl")

APPROVED = object()
REJECTED = object()
TIMEOUT = object()


class HITLManager:
    def __init__(self, webhook_url: Optional[str] = None, default_timeout: int = 3600):
        self.webhook_url = webhook_url or os.environ.get("HITL_WEBHOOK_URL", "")
        self.default_timeout = default_timeout
        self._pending: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, object] = {}

    async def request_approval(self, workflow_id: str, reason: str,
                               timeout: Optional[int] = None) -> tuple[object, str]:
        effective_timeout = timeout or self.default_timeout

        if self.webhook_url:
            self._send_webhook(workflow_id, reason)

        logger.info("HITL: workflow %s paused — %s (timeout=%ds)", workflow_id, reason, effective_timeout)

        event = asyncio.Event()
        self._pending[workflow_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            self._cleanup(workflow_id)
            return TIMEOUT, "approval timeout"

        decision = self._decisions.pop(workflow_id, REJECTED)
        self._cleanup(workflow_id)

        if decision is APPROVED:
            return APPROVED, "approved"
        return REJECTED, "rejected"

    def resolve(self, workflow_id: str, approved: bool) -> bool:
        if workflow_id not in self._pending:
            return False
        self._decisions[workflow_id] = APPROVED if approved else REJECTED
        self._pending[workflow_id].set()
        return True

    def _send_webhook(self, workflow_id: str, reason: str) -> None:
        payload = json.dumps({"workflow_id": workflow_id, "reason": reason, "action": "request_approval"}).encode()
        try:
            req = Request(self.webhook_url, data=payload, headers={"Content-Type": "application/json"})
            urlopen(req, timeout=5)
        except Exception as e:
            logger.warning("HITL webhook failed: %s", e)

    def _cleanup(self, workflow_id: str) -> None:
        self._pending.pop(workflow_id, None)
        self._decisions.pop(workflow_id, None)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

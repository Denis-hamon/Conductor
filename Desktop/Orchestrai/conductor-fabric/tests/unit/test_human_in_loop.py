"""Tests for HITLManager — approval webhooks and async resolution."""

import asyncio
import pytest

from runtime.human_in_loop import HITLManager, APPROVED, REJECTED, TIMEOUT


@pytest.fixture
def hitl():
    return HITLManager(webhook_url="")


@pytest.mark.asyncio
class TestHITLManager:
    async def test_request_approval_returns_approved(self, hitl):
        async def approve():
            await asyncio.sleep(0.01)
            hitl.resolve("wf-1", True)

        async def runner():
            return await hitl.request_approval("wf-1", "OK?", timeout=5)

        task = asyncio.create_task(approve())
        decision, reason = await runner()
        await task
        assert decision is APPROVED
        assert reason == "approved"

    async def test_request_approval_returns_rejected(self, hitl):
        async def reject():
            await asyncio.sleep(0.01)
            hitl.resolve("wf-2", False)

        async def runner():
            return await hitl.request_approval("wf-2", "OK?", timeout=5)

        task = asyncio.create_task(reject())
        decision, reason = await runner()
        await task
        assert decision is REJECTED
        assert reason == "rejected"

    async def test_request_approval_times_out(self, hitl):
        decision, reason = await hitl.request_approval("wf-timeout", "OK?", timeout=0.05)
        assert decision is TIMEOUT
        assert "timeout" in reason

    async def test_resolve_unknown_workflow_returns_false(self, hitl):
        result = hitl.resolve("nonexistent", True)
        assert result is False

    async def test_resolve_returns_true(self, hitl):
        async def resolve_after():
            await asyncio.sleep(0.01)
            result = hitl.resolve("wf-3", True)
            assert result is True

        async def runner():
            return await hitl.request_approval("wf-3", "OK?", timeout=5)

        task = asyncio.create_task(resolve_after())
        decision, reason = await runner()
        await task
        assert decision is APPROVED

    async def test_pending_count(self, hitl):
        async def count_after():
            await asyncio.sleep(0.01)
            assert hitl.pending_count == 1
            hitl.resolve("wf-4", True)

        async def runner():
            return await hitl.request_approval("wf-4", "OK?", timeout=5)

        task = asyncio.create_task(count_after())
        await runner()
        await task
        assert hitl.pending_count == 0

    async def test_cleanup_on_timeout(self, hitl):
        await hitl.request_approval("wf-cleanup", "OK?", timeout=0.05)
        assert hitl.pending_count == 0

    async def test_multiple_concurrent_approvals(self, hitl):
        async def resolve_all():
            await asyncio.sleep(0.02)
            hitl.resolve("wf-a", True)
            hitl.resolve("wf-b", True)

        async def runner_a():
            return await hitl.request_approval("wf-a", "A?", timeout=5)

        async def runner_b():
            return await hitl.request_approval("wf-b", "B?", timeout=5)

        resolver = asyncio.create_task(resolve_all())
        r_a, r_b = await asyncio.gather(runner_a(), runner_b())
        await resolver
        assert r_a[0] is APPROVED
        assert r_b[0] is APPROVED

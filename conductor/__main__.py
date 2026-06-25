"""Conductor Fabric — Heuristic Conductor entry point."""

import http.server
import json
import os
import logging

from conductor.router import classify_request
from conductor.planner import WorkflowPlanner
from conductor.fallback import FallbackPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conductor")

planner = WorkflowPlanner()
fallback = FallbackPlanner()

try:
    from training.phase0_cpt.model.cpt_planner import CptPlanner
    cpt_planner = CptPlanner()
    if cpt_planner.available:
        logger.info("CPT planner available")
    else:
        logger.info("CPT planner not available (no model path configured)")
        cpt_planner = None
except ImportError:
    logger.debug("CPT planner not installed")
    cpt_planner = None

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


class ConductorHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/conductor/plan":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length > 1_000_000:
            self._json(413, {"error": "request too large"})
            return
        body = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        messages = req.get("messages", [])
        if not messages:
            self._json(400, {"error": "messages required"})
            return

        content = messages[-1].get("content", "")

        route = classify_request(content)

        cpt_result = None
        if cpt_planner is not None:
            try:
                cpt_result = cpt_planner.generate(content)
            except Exception:
                logger.exception("CPT planner failed, falling back to heuristic")

        try:
            if cpt_result is not None and not cpt_result.fallback_used:
                plan = cpt_result.plan
                route = route._replace(
                    domain=cpt_result.plan.get("domain", route.domain),
                    reason=f"cpt_confidence={cpt_result.confidence:.2f}",
                )
            elif route.confidence >= 0.6:
                plan = planner.generate(route.domain, content)
            else:
                plan = planner.generate("general", content)
        except Exception:
            logger.exception("Conductor failed, activating fallback")
            plan = fallback.generate(content)
            route = route._replace(domain="general", reason="fallback activated")

        self._json(200, {"route": route._asdict(), "plan": plan})

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        logger.info(fmt % args)


def main():
    port = int(os.environ.get("CONDUCTOR_PORT", 9090))
    server = http.server.HTTPServer(("0.0.0.0", port), ConductorHandler)
    logger.info("Conductor listening on port %d", port)
    server.serve_forever()


if __name__ == "__main__":
    main()

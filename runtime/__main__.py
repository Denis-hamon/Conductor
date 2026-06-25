"""Conductor Fabric — Runtime entry point."""

import asyncio
import json
import logging
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

from runtime.engine import WorkflowEngine
from runtime.models import validate_plan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runtime")

engine = WorkflowEngine()


class RuntimeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/runtime/execute":
            self._handle_execute()
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"status": "ok"})
        elif self.path == "/readyz":
            self._json(503, {"status": "unavailable", "reason": "dependencies not checked"})
        else:
            self._json(404, {"error": "not found"})

    def _handle_execute(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 1_000_000:
            self._json(413, {"error": "request too large"})
            return
        body = self.rfile.read(length) if length else b"{}"
        try:
            plan = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        try:
            validate_plan(plan)
        except Exception as e:
            self._json(400, {"error": f"invalid plan: {e}"})
            return

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self._json(200, {"workflow_id": id(plan), "result": result})

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        logger.info(fmt % args)


def main():
    port = int(os.environ.get("RUNTIME_PORT", 7070))
    server = HTTPServer(("0.0.0.0", port), RuntimeHandler)
    logger.info("Runtime listening on port %d", port)
    server.serve_forever()


if __name__ == "__main__":
    main()

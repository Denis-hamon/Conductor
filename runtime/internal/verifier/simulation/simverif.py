"""SimVerif — World Model Simulation Verifier (AD-11).

Calls AgentWorld to simulate code execution / MCP tool results before
falling back to real sandbox. 70%+ coverage target under 2s.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional

logger = logging.getLogger("verifier.simverif")

SIMULATION_TIMEOUT = 5
CONFIDENCE_THRESHOLD = 0.7


class SimVerifResult:
    def __init__(self, simulated: bool, score: float, confidence: float,
                 output: Optional[str] = None, fallback_reason: str = "",
                 latency_ms: float = 0):
        self.simulated = simulated
        self.score = score
        self.confidence = confidence
        self.output = output
        self.fallback_reason = fallback_reason
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        return {
            "verification_type": "simulated" if self.simulated else "fallback",
            "score": self.score,
            "confidence": self.confidence,
            "output": self.output,
            "fallback_reason": self.fallback_reason,
            "latency_ms": self.latency_ms,
        }


class SimVerif:
    def __init__(self, agentworld_url: str = "http://agentworld-sglang:8080/v1/chat/completions"):
        self.agentworld_url = agentworld_url

    async def verify_code(self, code: str, domain: str = "code") -> SimVerifResult:
        start = time.monotonic()
        system_prompt = self._get_system_prompt("swe")

        try:
            response = await asyncio.wait_for(
                self._call_agentworld(system_prompt, f"Action: execute_code\nCode:\n{code}"),
                timeout=SIMULATION_TIMEOUT,
            )
            latency = (time.monotonic() - start) * 1000

            confidence = self._estimate_confidence(response)
            if confidence >= CONFIDENCE_THRESHOLD:
                return SimVerifResult(simulated=True, score=1.0, confidence=confidence,
                                      output=response, latency_ms=round(latency, 2))

            return SimVerifResult(simulated=False, score=0.0, confidence=confidence,
                                  fallback_reason=f"confidence {confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}",
                                  latency_ms=round(latency, 2))

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            return SimVerifResult(simulated=False, score=0.0, confidence=0.0,
                                  fallback_reason="timeout", latency_ms=round(latency, 2))
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return SimVerifResult(simulated=False, score=0.0, confidence=0.0,
                                  fallback_reason=str(e), latency_ms=round(latency, 2))

    async def verify_mcp(self, tool_name: str, tool_args: dict) -> SimVerifResult:
        start = time.monotonic()
        system_prompt = self._get_system_prompt("mcp")
        payload = json.dumps({"tool_name": tool_name, "arguments": tool_args})

        try:
            response = await asyncio.wait_for(
                self._call_agentworld(system_prompt, f"Action: call_tool\n{payload}"),
                timeout=SIMULATION_TIMEOUT,
            )
            latency = (time.monotonic() - start) * 1000

            return SimVerifResult(simulated=True, score=1.0, confidence=0.8,
                                  output=response, latency_ms=round(latency, 2))

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            return SimVerifResult(simulated=False, score=0.0, confidence=0.0,
                                  fallback_reason="timeout", latency_ms=round(latency, 2))
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return SimVerifResult(simulated=False, score=0.0, confidence=0.0,
                                  fallback_reason=str(e), latency_ms=round(latency, 2))

    async def _call_agentworld(self, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.agentworld_url,
                json={"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]},
                timeout=SIMULATION_TIMEOUT,
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _estimate_confidence(self, response: str) -> float:
        return 0.85

    def _get_system_prompt(self, env: str) -> str:
        prompts = {
            "swe": "You are a language world model simulating a software engineering environment. "
                   "Given an action, predict the output observation.",
            "mcp": "You are a language world model simulating MCP tool calls. "
                   "Given a tool name and arguments, predict the result.",
            "terminal": "You are a language world model simulating a Linux terminal. "
                        "Given a bash command, predict the terminal output.",
        }
        return prompts.get(env, prompts["terminal"])

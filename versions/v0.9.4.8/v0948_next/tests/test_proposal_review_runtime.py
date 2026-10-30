"""Exercise the real October proposal-review branch with offline HTTP models."""
import asyncio
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from gateway_server import configure_managed_d12_only
from v0948_next.gateway_integration import GenericMOAOrchestrator, ModelEndpoint
from v0948_next.proposal_review import MEMO_INSTRUCTION
from v0948_next.task_identity import resolve_task_identity


ROOT = Path(__file__).resolve().parents[2]
MEMO = "NEXT ACTION: update the parser.\nVERIFIED STATE: input is UTF-8.\nUNRESOLVED: run its test."
CALL = {"id": "execute-1", "type": "function", "function": {
    "name": "shell", "arguments": '{"command":"run-test"}'}}
MESSAGES = [
    {"role": "system", "content": "Keep the existing files."},
    {"role": "user", "content": "Implement the parser; keep UTF-8 support."},
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "read-1", "type": "function", "function": {
            "name": "shell", "arguments": '{"command":"read-parser"}'}}]},
    {"role": "tool", "tool_call_id": "read-1", "content": "Observed parser: UTF-8."},
    {"role": "user", "content": "Keep completed work and run its test next."},
]


def completion(content="", *, calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = deepcopy(calls)
    return {"id": "offline", "object": "chat.completion", "choices": [
        {"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def message_text(message):
    content = message.get("content")
    return content if isinstance(content, str) else "\n".join(
        block.get("text", "") for block in content or [] if isinstance(block, dict))


class ProposalReviewRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.requests = []
        self.review_responses = {"D16": "NO_CHANGE", "D9": "NO_ALTERNATIVE", "D13": "STATE_OK"}
        self.memo_response = completion(MEMO)
        self.executor_responses = []
        self.slow = set()
        self.cancelled = set()
        self.cancellation_seen = asyncio.Event()

        async def handle(request):
            body = json.loads(request.content)
            self.requests.append((request.url.host, deepcopy(body)))
            actor = request.url.host
            if actor == "d12":
                if message_text(body["messages"][0]).startswith(MEMO_INSTRUCTION):
                    return httpx.Response(200, json=self.memo_response)
                if self.executor_responses:
                    status, payload = self.executor_responses.pop(0)
                    return httpx.Response(status, json=payload)
                return httpx.Response(200, json=completion(calls=[CALL], finish="tool_calls"))
            actor = actor.upper()
            if actor in self.slow:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    self.cancelled.add(actor)
                    if self.cancelled == self.slow:
                        self.cancellation_seen.set()
                    raise
            value = self.review_responses[actor]
            if isinstance(value, int):
                return httpx.Response(value, json={"error": {"message": "offline unavailable"}})
            return httpx.Response(200, json=completion(value))

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(self.client.aclose)
        self.orchestrator = GenericMOAOrchestrator(
            executor=ModelEndpoint("d12", "http://d12/v1"),
            analysts={actor: ModelEndpoint(actor, "http://" + actor.lower() + "/v1")
                      for actor in ("D16", "D9", "D13")},
            run_id="offline-test", trace_path=Path(self.tmp.name) / "trace.jsonl",
            client=self.client, panel_strategy="proposal_review",
            proposal_review_profiles=json.loads((ROOT / "proposal_review_profiles.json").read_text()),
            solver_profiles=json.loads((ROOT / "solver_profiles.json").read_text()),
            default_collaboration_mode="initial_panel", executor_max_tokens=16384,
            analyst_deadline_seconds=1,
        )

    def body(self, task_id="test-task"):
        return {"model": "fusion-moa-v0948-general", "messages": deepcopy(MESSAGES),
                "metadata": {"task_id": task_id, "moa": {"mode": "initial_panel"}},
                "tools": [{"type": "function", "function": {"name": "shell", "parameters": {
                    "type": "object", "properties": {"command": {"type": "string"}}}}}],
                "tool_choice": "auto", "max_tokens": 16384, "temperature": 0.6,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}

    async def run_request(self, body=None):
        body = body or self.body()
        return await self.orchestrator.complete(
            body, identity=resolve_task_identity(body["messages"], body["metadata"]),
            public_model=body["model"], managed=True)

    def executors(self):
        return [b for actor, b in self.requests if actor == "d12"
                and not message_text(b["messages"][0]).startswith(MEMO_INSTRUCTION)]

    async def test_memo_and_independent_reviewers_are_read_only_final_d12_keeps_tools_and_context(self):
        body = self.body()
        result = await self.run_request(body)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["choices"][0]["message"]["tool_calls"], [CALL])
        self.assertEqual([actor for actor, _ in self.requests], ["d12", "d16", "d9", "d13", "d12"])
        memo_body = self.requests[0][1]
        self.assertEqual(json.loads(memo_body["messages"][1]["content"])["execution_record"], MESSAGES)
        self.assertEqual(memo_body["max_tokens"], 768)
        self.assertFalse(memo_body["chat_template_kwargs"]["enable_thinking"])
        reviews = [b for actor, b in self.requests if actor != "d12"]
        for auxiliary in [memo_body, *reviews]:
            self.assertNotIn("tools", auxiliary)
            self.assertNotIn("tool_choice", auxiliary)
        self.assertEqual(len({b["messages"][1]["content"] for b in reviews}), 1)
        packet = json.loads(reviews[0]["messages"][1]["content"])
        self.assertEqual(packet["latest_user_request"], MESSAGES[-1])
        self.assertEqual(packet["d12_consult_memo"], MEMO)
        self.assertEqual(packet["latest_tool_exchange"]["context_index"], 3)
        self.assertEqual(packet["latest_tool_exchange"]["result"], MESSAGES[3])
        self.assertNotIn("reviews", packet)
        final = self.executors()[0]
        self.assertEqual([m for m in final["messages"] if m["role"] != "system"], MESSAGES[1:])
        self.assertIn(MESSAGES[0]["content"], message_text(final["messages"][0]))
        for key in ("tools", "tool_choice", "max_tokens", "temperature", "chat_template_kwargs"):
            self.assertEqual(final[key], body[key])
        self.assertEqual(body, self.body())

    async def test_initial_panel_is_once_per_task_and_explicit_recall_still_works(self):
        self.executor_responses = [(200, completion(calls=[CALL], finish="length"))]
        await self.run_request()
        await self.run_request()
        self.assertEqual(self.orchestrator.d12_proposal_request_count, 1)
        recall = self.body()
        recall["metadata"]["moa"]["mode"] = "panel_once"
        await self.run_request(recall)
        self.assertEqual(self.orchestrator.d12_proposal_request_count, 2)
        await self.run_request(self.body("other-task"))
        self.assertEqual(self.orchestrator.initial_panel_event_count, 2)

    async def test_partial_timeout_keeps_completed_review_under_one_deadline(self):
        self.orchestrator.analyst_deadline_seconds = 0.05
        self.review_responses["D16"] = "CHANGE: preserve UTF-8."
        self.slow = {"D9", "D13"}
        result = await self.run_request()
        self.assertEqual(result.status_code, 200)
        guidance = message_text(self.executors()[0]["messages"][0])
        self.assertIn("CHANGE: preserve UTF-8.", guidance)
        opinions = json.loads(guidance.splitlines()[-1])["reviews"]
        self.assertEqual([item["actor"] for item in opinions], ["D16"])
        await asyncio.wait_for(self.cancellation_seen.wait(), timeout=0.5)
        self.assertEqual(self.cancelled, self.slow)
        self.assertEqual(self.orchestrator.d12_proposal_request_count, 1)

    async def test_zero_reviews_retains_completed_memo_and_continues(self):
        self.review_responses = {"D16": "", "D9": 503, "D13": ""}
        result = await self.run_request()
        self.assertEqual(result.status_code, 200)
        final = message_text(self.executors()[0]["messages"][0])
        self.assertIn(MEMO.splitlines()[0], final)
        self.assertIn('"reviews": []', final)

    async def test_invalid_memo_fails_open_without_dispatching_reviewers(self):
        self.memo_response = completion("unfinished", finish="length")
        result = await self.run_request()
        self.assertEqual(result.status_code, 200)
        self.assertEqual([actor for actor, _ in self.requests], ["d12", "d12"])
        self.assertNotIn("[MOA D12 proposal", message_text(self.executors()[0]["messages"][0]))

    async def test_advice_context_error_retries_full_base_without_compacting_it(self):
        self.executor_responses = [(400, {"error": {"message": "context length exceeded"}})]
        result = await self.run_request()
        self.assertEqual(result.status_code, 200)
        first, fallback = self.executors()
        self.assertIn("[MOA D12 proposal", message_text(first["messages"][0]))
        self.assertNotIn("[MOA D12 proposal", message_text(fallback["messages"][0]))
        self.assertEqual([m for m in fallback["messages"] if m["role"] != "system"], MESSAGES[1:])
        self.assertEqual(self.orchestrator.context_compaction_count, 0)

    async def test_managed_d12_only_keeps_output_recovery_and_disables_aux_calls(self):
        configure_managed_d12_only(self.orchestrator)
        self.executor_responses = [(200, completion("truncated reasoning", finish="length"))]
        result = await self.run_request()
        self.assertEqual(result.headers["X-MOA-Output-Contract"], "recovered")
        self.assertEqual([actor for actor, _ in self.requests], ["d12", "d12"])
        self.assertEqual(self.orchestrator.d12_proposal_request_count, 0)
        recovery = self.executors()[1]
        self.assertEqual(recovery["max_tokens"], 512)
        self.assertEqual(recovery["messages"][0]["role"], "system")
        self.assertEqual([m for m in recovery["messages"] if m["role"] != "system"], MESSAGES[1:])
        self.assertEqual(recovery["tools"], self.body()["tools"])
        compacted, detail = await self.orchestrator._compact_context(MESSAGES)
        self.assertIsNone(compacted)
        self.assertEqual(detail["analyst_calls"], 0)

    async def test_managed_d12_only_retains_hard_episode_clean_recovery(self):
        configure_managed_d12_only(self.orchestrator)
        body = self.body()
        body["metadata"]["moa"] = {"mode": "auto", "hard_stall": True}
        repeated = deepcopy(MESSAGES[2]["tool_calls"])
        self.executor_responses = [(200, completion(calls=repeated, finish="tool_calls"))]
        result = await self.run_request(body)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["X-MOA-Recovery"], "changed_action")
        self.assertEqual(result.payload["choices"][0]["message"]["tool_calls"], [CALL])
        self.assertEqual([actor for actor, _ in self.requests], ["d12", "d12"])
        self.assertEqual(self.orchestrator.recovery_generation_count, 1)
        self.assertEqual(self.orchestrator.d12_proposal_request_count, 0)

    async def test_http_health_aliases_and_tool_stream_terminal(self):
        with patch.dict(os.environ, {"FUSION_TRACE_PATH": str(Path(self.tmp.name) / "api.jsonl")}):
            gateway = importlib.import_module("fusion_terminal_server_v0948")
        await gateway.orchestrator.aclose()
        transport = httpx.ASGITransport(app=gateway.app)
        with patch.object(gateway, "orchestrator", self.orchestrator):
            async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
                health = await client.get("/health")
                self.assertEqual(health.json()["version"], "v0.9.4.8")
                models = (await client.get("/v1/models")).json()["data"]
                self.assertIn("fusion-moa-v0942-general", {m["id"] for m in models})
                body = self.body()
                body.update(stream=True)
                response = await client.post("/v1/chat/completions", json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["x-fusion-moa-version"], "v0.9.4.8")
                events = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                native_calls = [call for event in events for choice in event.get("choices", [])
                                for call in choice.get("delta", {}).get("tool_calls", [])]
                self.assertEqual(native_calls[0]["function"]["name"], "shell")
                self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))


if __name__ == "__main__":
    unittest.main()

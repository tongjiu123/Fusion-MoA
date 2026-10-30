"""Offline callback contract checks without requiring the optional bridge runtime."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


BRIDGE = Path(__file__).resolve().parents[2] / "adapters/claude_bridge"


def load(name):
    spec = importlib.util.spec_from_file_location("offline_" + name, BRIDGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BridgeControlTests(unittest.TestCase):
    def callback(self):
        # Stub only LiteLLM's base logger; exercise the real callback logic.
        logger = types.ModuleType("litellm.integrations.custom_logger")
        logger.CustomLogger = object
        with patch.dict(sys.modules, {"litellm.integrations.custom_logger": logger}), \
             patch.dict(os.environ, {"FUSION_TERMINAL_COMPAT": "0"}):
            return load("callback").AnthropicHeartbeat()

    def test_task_id_explicit_body_wins_and_empty_body_does_not_fall_back(self):
        handler = self.callback()
        for value in ("body-task", ""):
            outgoing = {"messages": [{"role": "user", "content": "hello"}]}
            kwargs = {"litellm_params": {"custom_llm_provider": "openai", "proxy_server_request": {
                "body": {"metadata": {"task_id": value}}, "headers": {"X-Fusion-Task-ID": "header-task"}}},
                "additional_args": {"complete_input_dict": outgoing}}
            with patch.dict(os.environ, {"FUSION_TERMINAL_COMPAT": "0"}):
                handler.log_pre_api_call("model", [], kwargs)
            self.assertEqual(outgoing.get("metadata"), {"task_id": value} if value else None)

    def test_header_task_id_is_forwarded_without_inventing_identity(self):
        handler = self.callback()
        outgoing = {"messages": [], "metadata": {"other": "keep"}}
        kwargs = {"litellm_params": {"custom_llm_provider": "openai", "proxy_server_request": {
            "body": {}, "headers": {"x-fusion-task-id": "header-task"}}},
            "additional_args": {"complete_input_dict": outgoing}}
        with patch.dict(os.environ, {"FUSION_TERMINAL_COMPAT": "0"}):
            handler.log_pre_api_call("model", [], kwargs)
        self.assertEqual(outgoing["metadata"], {"other": "keep", "task_id": "header-task"})

    def test_reasoning_history_preserves_tools_and_fills_only_missing_content(self):
        normalize = load("stream_compat").normalize_reasoning_history
        messages = [{"role": "assistant", "thinking_blocks": [{"type": "thinking", "thinking": "reason"}]},
                    {"role": "assistant", "reasoning_content": "reason", "tool_calls": [{"id": "call"}]},
                    {"role": "user", "content": "continue"}]
        result = normalize(messages)
        self.assertEqual(result[0]["reasoning_content"], "reason")
        self.assertEqual(result[0]["content"], "")
        self.assertEqual(result[1:], messages[1:])
        self.assertNotIn("content", messages[0])


if __name__ == "__main__":
    unittest.main()

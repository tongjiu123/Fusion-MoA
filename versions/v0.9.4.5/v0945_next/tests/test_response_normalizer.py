from __future__ import annotations

import unittest

from v0945_next.response_normalizer import (
    ThinkingStreamNormalizer,
    normalize_completion_payload,
    split_leading_thinking,
    split_unclosed_leading_thinking,
)


def stream_chunk(content=None, *, finish_reason=None):
    delta = {} if content is None else {"content": content}
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "fusion-moa-v0942-general",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


class ThinkingResponseNormalizerTests(unittest.TestCase):
    def test_complete_leading_block_is_split(self) -> None:
        final, reasoning, changed = split_leading_thinking(
            "\n<think>inspect the evidence</think>\nfinal answer"
        )
        self.assertTrue(changed)
        self.assertEqual(reasoning, "inspect the evidence")
        self.assertEqual(final, "final answer")

    def test_nonleading_or_malformed_markup_fails_open(self) -> None:
        for content in ("show the literal <think> tag", "<thi"):
            final, reasoning, changed = split_leading_thinking(content)
            self.assertFalse(changed)
            self.assertEqual(final, content)
            self.assertEqual(reasoning, "")

    def test_unclosed_leading_block_is_separated_at_terminal(self) -> None:
        final, reasoning, changed = split_unclosed_leading_thinking(
            "\n<think>unfinished reasoning"
        )
        self.assertTrue(changed)
        self.assertEqual(final, "")
        self.assertEqual(reasoning, "unfinished reasoning")

        normalized = normalize_completion_payload(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>unfinished reasoning",
                        },
                        "finish_reason": "length",
                    }
                ]
            }
        )
        message = normalized["choices"][0]["message"]
        self.assertEqual(message["content"], "")
        self.assertEqual(message["reasoning_content"], "unfinished reasoning")

    def test_payload_preserves_tools_and_moves_reasoning(self) -> None:
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>plan</think>\nrun the tool",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        normalized = normalize_completion_payload(payload)
        message = normalized["choices"][0]["message"]
        self.assertEqual(message["content"], "run the tool")
        self.assertEqual(message["reasoning_content"], "plan")
        self.assertEqual(
            message["tool_calls"],
            payload["choices"][0]["message"]["tool_calls"],
        )
        self.assertNotIn("reasoning_content", payload["choices"][0]["message"])

    def test_streaming_block_can_span_chunks(self) -> None:
        normalizer = ThinkingStreamNormalizer()
        outputs = []
        for content in ("<thi", "nk>secret", "</thi", "nk>\nanswer"):
            outputs.extend(normalizer.feed(stream_chunk(content)))
        outputs.extend(normalizer.feed(stream_chunk(finish_reason="stop")))

        deltas = [item["choices"][0]["delta"] for item in outputs]
        self.assertEqual(
            "".join(delta.get("reasoning_content", "") for delta in deltas),
            "secret",
        )
        self.assertEqual(
            "".join(delta.get("content", "") for delta in deltas),
            "answer",
        )
        self.assertNotIn("<think>", str(outputs))
        self.assertEqual(outputs[-1]["choices"][0]["finish_reason"], "stop")

    def test_unclosed_streaming_block_is_separated_at_terminal(self) -> None:
        normalizer = ThinkingStreamNormalizer()
        outputs = normalizer.feed(stream_chunk("<think>unfinished"))
        self.assertEqual(outputs, [])
        outputs.extend(normalizer.feed(stream_chunk(finish_reason="length")))
        content = "".join(
            item["choices"][0]["delta"].get("content", "")
            for item in outputs
        )
        reasoning = "".join(
            item["choices"][0]["delta"].get("reasoning_content", "")
            for item in outputs
        )
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "unfinished")
        self.assertNotIn("<think>", str(outputs))
        self.assertEqual(outputs[-1]["choices"][0]["finish_reason"], "length")

    def test_unclosed_streaming_block_is_separated_on_transport_end(self) -> None:
        normalizer = ThinkingStreamNormalizer()
        self.assertEqual(
            normalizer.feed(stream_chunk("<think>partial transport")),
            [],
        )
        outputs = normalizer.flush()
        delta = outputs[0]["choices"][0]["delta"]
        self.assertEqual(delta["reasoning_content"], "partial transport")
        self.assertNotIn("<think>", str(outputs))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from v0945_next.evidence_episode import StallDecision
from v0945_next.gateway_base import GenericMOAOrchestrator as BaseOrchestrator
from v0945_next.gateway_integration import GenericMOAOrchestrator
from v0945_next.request_controls import normalize_request_body
from v0945_next.soft_stall import (
    PANEL_ONCE_REASON,
    evaluate_collaboration_gate_v0945,
)


class PanelOnceRequestTests(unittest.TestCase):
    def test_top_level_control_is_normalized_and_not_forwarded(self) -> None:
        body = normalize_request_body(
            {
                "model": "fusion-moa-v0944-general",
                "messages": [{"role": "user", "content": "2+2?"}],
                "moa": {"mode": "panel_once"},
            }
        )
        self.assertNotIn("moa", body)
        self.assertEqual(body["metadata"]["moa"]["mode"], "panel_once")
        upstream = BaseOrchestrator._upstream_body(
            body,
            model="d12",
            messages=body["messages"],
        )
        self.assertNotIn("moa", upstream)
        self.assertNotIn("metadata", upstream)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "moa.mode"):
            normalize_request_body(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "moa": {"mode": "unknown"},
                }
            )

    def test_panel_once_is_explicit_and_unique_per_request(self) -> None:
        messages = [{"role": "user", "content": "solve this once"}]
        metadata = {"moa": {"mode": "panel_once"}}
        first = evaluate_collaboration_gate_v0945(
            messages,
            metadata,
            request_nonce="request-a",
        )
        second = evaluate_collaboration_gate_v0945(
            messages,
            metadata,
            request_nonce="request-b",
        )
        self.assertTrue(first.admitted)
        self.assertEqual(first.reason, PANEL_ONCE_REASON)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_auto_and_disabled_preserve_parent_behavior(self) -> None:
        messages = [{"role": "user", "content": "ordinary request"}]
        automatic = evaluate_collaboration_gate_v0945(
            messages,
            {"moa": {"mode": "auto"}},
            request_nonce="unused",
        )
        disabled = evaluate_collaboration_gate_v0945(
            messages,
            {"moa": {"mode": "panel_once", "enabled": False}},
            request_nonce="unused",
        )
        self.assertFalse(automatic.admitted)
        self.assertEqual(automatic.reason, "no_structural_stall")
        self.assertFalse(disabled.admitted)
        self.assertEqual(disabled.reason, "caller_disabled")

    def test_panel_once_deliberation_preserves_original_messages(self) -> None:
        messages = [{"role": "user", "content": "write a solution"}]
        stall = StallDecision(
            admitted=True,
            reason=PANEL_ONCE_REASON,
            fingerprint="f",
            episode_id="e",
        )
        forwarded = GenericMOAOrchestrator._d12_deliberation_messages(
            messages,
            messages,
            stall=stall,
            integrated_brief="brief text",
        )
        self.assertEqual(forwarded[0], messages[0])
        self.assertIn("explicit single-turn", forwarded[-1]["content"])
        self.assertIn("brief text", forwarded[-1]["content"])


if __name__ == "__main__":
    unittest.main()

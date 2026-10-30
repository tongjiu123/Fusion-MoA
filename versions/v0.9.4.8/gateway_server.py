"""Portable launcher for the published runtime; no experiment wire recording."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parent


async def consult_without_aux(self, body, executor_base, raw_messages, *,
                              public_model, task_id, deadline):
    """Same no-op as the recorded managed_no_memo runtime branch."""
    return "", {
        "attempted": False, "request_count": 0,
        "selected_hypothesis_ids": [], "status": "managed_no_memo",
        "review": {
            "strategy": "proposal_review", "status": "managed_no_memo",
            "proposal": {"attempted": False}, "candidates": [], "available_count": 0,
        },
    }


async def compact_without_aux(self, messages):
    """Keep managed execution history; do not call auxiliary Compact models."""
    return None, {
        "status": "aux_unavailable", "analyst_calls": 0,
        "reason": "auxiliary_compact_disabled_for_managed_d12_only_baseline",
    }


def configure_managed_d12_only(orchestrator):
    # Preserve the same admission/episode/Recovery implementation. In particular,
    # setting managed=False would also disable Recovery and change the control.
    if orchestrator.panel_strategy != "proposal_review":
        raise ValueError("managed D12-only requires the proposal_review configuration")
    orchestrator._consult_review = types.MethodType(consult_without_aux, orchestrator)
    orchestrator._compact_context = types.MethodType(compact_without_aux, orchestrator)
    orchestrator.boundaries["publication_mode"] = "managed_d12_only"
    orchestrator.boundaries["auxiliary_compact_enabled"] = False


def load_gateway(config: Path, mode: str):
    values = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                           for k, v in values.items()):
        raise ValueError("gateway config must map environment names to strings")
    for key, value in values.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("FUSION_PORT", "19036")
    os.environ.setdefault("FUSION_RUN_ID", "v0948-" + mode)
    gateway = importlib.import_module("fusion_terminal_server_v0948")
    if mode == "d12-only":
        configure_managed_d12_only(gateway.orchestrator)
        # The historical *-solo-d12 aliases mean managed=False. This launcher
        # exposes general aliases only, all with the recorded managed behavior.
        gateway.MODEL_POLICIES = {
            name: policy for name, policy in gateway.MODEL_POLICIES.items()
            if policy["managed"]
        }
    else:
        gateway.orchestrator.boundaries["publication_mode"] = "full_fusion"
    return gateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/gateway.environment.template.json")
    parser.add_argument("--mode", choices=("full", "d12-only"), default="full")
    args = parser.parse_args()
    gateway = load_gateway(args.config, args.mode)
    import uvicorn
    uvicorn.run(gateway.app, host="127.0.0.1", port=gateway.PORT)


if __name__ == "__main__":
    main()

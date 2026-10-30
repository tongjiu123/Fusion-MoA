# Fusion-MOA v0.9.4.8

This is the complete Gateway from the October 13, 2026 runtime snapshot identified
by the October 14 archive overview. The entry point, all `v0948_next/` runtime
modules, schemas and current review profiles are included. No earlier version is
imported at runtime. The September baseline and its hardware scripts are not used
as a substitute for this October snapshot.

The core files are preserved verbatim. Portable launchers, deployment documentation
and offline tests are publication additions. Internal source inventories and
experiment records are not part of this public package.

## Execution boundary

```text
External Agent: task loop, tool execution and evaluation
    │ OpenAI messages + tools, or Claude Messages through the included bridge
    ▼
Gateway: initial_panel + conditional recall
    ├─ D12: NEXT ACTION / VERIFIED STATE / UNRESOLVED Memo (read-only call)
    ├─ D16: implementation delta ┐
    ├─ D9: alternative path      ├─ independent reviews of the same packet
    └─ D13: state consistency   ┘
    ▼
D12: full effective execution context + valid Memo + completed reviews
    │ sole final text / native tool_calls
    ▼
External Agent executes tools and supplies actual observations
```

Memo is D12's own short, revisable action proposal. Its call receives the complete
effective execution record as read-only data, including any existing Compact;
it does not reconstruct history that Compact already replaced. The three reviewers
receive the same Memo, latest real user request and selected latest tool call/result
with its `context_index`. The user request comes from the incoming messages;
the tool exchange comes from the effective execution record. They do not receive
one another's opinions. No tools or
`tool_choice` are bound to Memo/reviewer API calls. D12's final call retains its
original history, roles, tools, sampling and output budget; advice does not replace
execution context. This branch consumes text opinions, not typed evidence packets
or a top-two selector, and has no Integrator or voting gate.

One shared 120-second absolute deadline covers Memo and parallel reviewers. The
collector waits until all reviews finish or that deadline; at the deadline,
completed partial opinions remain usable and unfinished generations are discarded. With
zero completed opinions, a valid Memo still reaches D12. If Memo is unavailable,
D12 continues normally. Advice that causes a context error is dropped before
retrying the original execution context. `NO_CHANGE`, `NO_ALTERNATIVE` and `STATE_OK`
are valid no-op responses, not proofs of correctness or new information.

The first request of an opaque task ID uses `initial_panel`; later requests use
existing Soft Stall, Hard/Exact/Result Cycle or explicit `panel_once` recall.
`always` remains an explicit mode. Initial state is process-local; restarts do not
provide persistent session deduplication. Task lists, tool execution, environment
creation, retries across attempts and official scoring belong to external agents.

## Start the Gateway

Use Python 3.13 for the verified environment. Four separately operated
OpenAI-compatible model services must already be available. This package contains
neither model weights nor an inference engine or GPU provisioning scripts.

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r versions/v0.9.4.8/requirements.txt
cp versions/v0.9.4.8/config/gateway.environment.template.json versions/v0.9.4.8/config/gateway.local.json
bash versions/v0.9.4.8/run_gateway.sh --config config/gateway.local.json
```

Edit the local JSON to match your endpoints and their advertised model names.
Its values are environment defaults; an already set environment variable wins.
`run_gateway.sh` changes into this package before reading relative paths. The
default listener is `http://127.0.0.1:19036`; set `FUSION_PORT` to change its port.
Default profiles resolve relative to the package. Use `FUSION_REVIEW_CONFIG` and
`FUSION_SOLVER_CONFIG` for alternatives. The direct original
`fusion_terminal_server_v0948.py` entry retains its historical defaults (including
`legacy` strategy); use the launcher and supplied config for the October profile.

| Cell | Recorded October model service | Configuration |
| --- | --- | --- |
| D12 | Qwen3.8-Flash-Next IQ3_S | final execution: thinking medium, temperature 0.6, output 16,384; Memo: non-thinking, max 768 |
| D16 | Qwopus3.8-27B-Flash-V2 Q8_0 | implementation review: non-thinking, temperature 0.3, max 768 |
| D9 | Qwopus3.6-27B-v2 Q6_K | alternative review: non-thinking, temperature 0.3, max 512 |
| D13 | Muse-Glimmer-30B Dynamic Q4_K_XL | state review: native `reasoning_strength=low`, temperature 0.3, max 512 |

The five-GPU allocation and quantizations describe that run, not a universal
hardware requirement. D12 and D9 did not enable MTP in that run; D16 used MTP
draft n=4. `proposal_review_profiles.json` contains the current review settings.
`solver_profiles.json` preserves the different legacy solver settings and is not
the active reviewer configuration. The gateway environment does not override
the caller's final D12 sampling; use the request example or bridge profile below.

```bash
curl -sS http://127.0.0.1:19036/health
curl -sS http://127.0.0.1:19036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fusion-moa-v0948-general","messages":[{"role":"user","content":"Explain the next step."}],"metadata":{"task_id":"example-task-1"},"max_tokens":16384,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"medium"}}'
```

Reuse `metadata.task_id` for all requests in one logical task. Use a new ID for an
independent attempt. Verify `/health`, `run_id` and `X-Fusion-MOA-Version`; accepted
old model aliases do not mean an old implementation is executing. The original
core accepts v0948 through v0942 general/solo aliases. Runtime trace output goes to
the package's ignored `runs/` directory by default; choose `FUSION_RUNS_DIR` or
`FUSION_TRACE_PATH` locally. Runtime records are not part of this source release.

## Managed D12-only control

```bash
bash versions/v0.9.4.8/run_gateway.sh --mode d12-only --config config/gateway.local.json
```

Use `fusion-moa-v0948-general` for this control. It shares the complete managed
execution, task admission/episode state and Recovery logic; only the consult call
and auxiliary Compact are disabled using the recorded control's no-op behavior.
It does not set `managed=False`. The older `*-solo-d12` aliases have that historical
unmanaged meaning, so this launcher does not expose them in D12-only mode. A
general request with `moa.enabled=false` also does not reproduce the same control,
because it bypasses collaboration admission and changes associated lifecycle behavior.

## Claude Code bridge

The included adapter is the external LiteLLM 1.93.0 bridge from the same October
snapshot. It adds Anthropic SSE heartbeat, task ID forwarding, mixed
reasoning/text/tool-delta handling and thinking-history normalization. Its profile
uses the Gateway's Chat Completions route and medium/0.6 D12 execution settings.

```bash
python -m pip install -r versions/v0.9.4.8/adapters/claude_bridge/requirements.txt
bash versions/v0.9.4.8/adapters/claude_bridge/run_bridge.sh
```

Point the Claude protocol client at `http://127.0.0.1:19037`. The bridge's
`litellm.yaml` points to the loopback Gateway with a placeholder key. To change it,
make a local YAML copy and set `LITELLM_CONFIG`. Keep `FUSION_TERMINAL_COMPAT=1`
for this pinned bridge; the launcher sets it. The adapter hooks private LiteLLM
interfaces, so a different LiteLLM version requires requalification.

Forward a task ID with `X-Fusion-Task-ID` (for example, through the client's
`ANTHROPIC_CUSTOM_HEADERS`). An explicit body `metadata.task_id` takes precedence,
including an invalid/empty explicit value; the bridge does not invent an ID from
an account, model or task title. The recorded external runtime used Claude Code
2.1.220 and Harbor 0.15.0; those agents and benchmark resources are not bundled.
Other clients and modalities require their own compatibility checks.

## Offline verification

From this package directory, with its dependencies installed:

```bash
python -m unittest discover -s v0948_next/tests -v
python -m compileall -q fusion_terminal_server_v0948.py gateway_server.py v0948_next adapters
bash -n run_gateway.sh
bash -n adapters/claude_bridge/run_bridge.sh
```

Tests exercise the real orchestrator through `httpx.MockTransport`, including
read-only Memo/reviews, unchanged final tools/context, first-call plus explicit
recall, partial timeout cancellation, zero reviews, invalid Memo, advice context
fallback, and managed D12-only Clean/Output Recovery. Inherited response normalization
and panel control tests are retained; one stale pre-system-first test expectation
is corrected without changing its runtime implementation. These checks do not
run model inference, GPU deployment, an external agent or an official benchmark.

The runtime requires only the existing packages in `requirements.txt`; the bridge
adds its existing pinned LiteLLM dependency. The root `pyproject.toml` continues to
describe the separate minimal R1 API. This release adds no license declaration.

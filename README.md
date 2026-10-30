# Fusion-MOA

This repository contains the original minimum viable Pioneer R1 framework and
the separately versioned, self-contained **Fusion-MOA v0.9.4.8 Gateway**.

Development repository:
[`tongjiu123/Fusion-MoA`](https://github.com/tongjiu123/Fusion-MoA).
The original [`Fusion-MoA-Pioneer-R1`](https://github.com/tongjiu123/Fusion-MoA-Pioneer-R1)
is retained separately as the historical R1 archive; it is not renamed or replaced.

| Source | Purpose | Status |
| --- | --- | --- |
| [`fusion_moa/`](fusion_moa/) | Minimal Pioneer R1 Python framework | Original API and package version retained |
| [`versions/v0.9.4.8/`](versions/v0.9.4.8/README.md) | Complete Gateway, current profiles, Claude protocol bridge, portable launchers and offline tests | Latest source release, from the complete October 13 runtime snapshot |
| [`versions/v0.9.4.5/`](versions/v0.9.4.5/README.md) | Earlier Gateway candidate | Historical source retained |

## Complete Gateway: v0.9.4.8

**通用、单入口、OpenAI-compatible、D12 唯一最终负责、MoA 按需辅助。**

The Gateway accepts OpenAI-compatible `messages + tools`. D12 generates a short
three-part Memo: **NEXT ACTION / VERIFIED STATE / UNRESOLVED**. D16, D9 and D13
independently review the same proposal and selected observed evidence. They have
no executable tools and do not see one another's opinions. D12 keeps its complete
effective execution context, including an existing Compact and subsequent
messages, and produces the only final text or native `tool_calls`.

The published launcher uses `initial_panel + proposal_review`: consult on the
first request of a task, then recall through the existing stall signals or an
explicit request. Memo and parallel reviews share one 120-second deadline.
The collector waits for all reviews or the deadline; completed partial reviews
remain usable at that deadline. Zero reviews retain a valid Memo, and
an unavailable Memo falls back to normal D12 execution. D12 can adopt part of an
opinion, reject it or keep its plan. This path uses text reviews; the older typed
packet/ID selection and Integrator path is retained for compatibility and is
not an active guarantee of proposal review.

External agents execute tools, maintain their task loops and run scoring. The
Gateway does not contain benchmark tasks, task-specific routing or a scorer.
The managed D12-only control keeps execution management and Recovery while
disabling Memo, expert calls and auxiliary Compact; it is not a bare model.

See the [runnable package and deployment instructions](versions/v0.9.4.8/README.md)
and [architecture and publication scope](docs/v0.9.4.8.md).
The release preserves the real Gateway implementation rather than replacing it
with the small Python API below. Model weights, third-party inference engines,
credentials, private deployment records, logs and benchmark task data are excluded.
Publishing this source does not switch a running service.

## Benchmark results

Evaluation dataset: **Terminal-Bench 2.1**, 89 tasks, using the original official
verifier through the external evaluation harness. Only aggregate results are
published; per-task records, raw scoring files, trajectories and audit artifacts
remain private.

| System and evaluation condition | PASS | FAIL | UNSCORED | Tasks |
| --- | ---: | ---: | ---: | ---: |
| Managed D12-only, original per-task limits | 62 | 25 | 2 | 89 |
| Full Fusion, original per-task limits | 60 | 29 | 0 | 89 |
| Full Fusion, retained coverage with longer budgets and repeated attempts | 76 | 13 | 0 | 89 |
| LongCat-2.0 + Terminus-2, historical run | 45 | 42 | 2 | 89 |
| HY3 + Terminus-2, historical run | 51 | 36 | 2 | 89 |
| DeepSeek V4 Flash 0731 + Terminus-2, historical run | 61 | 27 | 1 | 89 |

The 76/89 coverage includes additional attempts at 3× or 5× the original Agent
time limits; it is **not a single-attempt or equal-budget improvement** over
D12-only. At the original limits, Full Fusion did not exceed D12-only.
The historical external runs used different agents and execution budgets, so
these rows are not an equal-condition model ranking. Missing official scores
remain UNSCORED, not FAIL. See the [evaluation conditions](benchmarks/terminal_bench_2_1/README.md).

## Original minimal Pioneer R1 framework

The original framework preserves the core mixture-of-agents (MoA) boundary:

1. a request is either handled directly or admitted to collective inference;
2. read-only model clients produce candidate evidence concurrently;
3. a configurable selector chooses which evidence to pass onward; and
4. one executor produces the only externally returned answer.

Compared with the complete Gateway, this minimal implementation removes
deployment-specific redundancy such as provider bindings, infrastructure
adapters, operations telemetry, and evaluation assets while retaining the core
collective-inference flow.

## Minimal example

```python
import asyncio

from fusion_moa import MoAFramework, Request


class Client:
    def __init__(self, name: str) -> None:
        self.name = name

    async def complete(self, messages):
        return f"{self.name}: candidate response"


async def main() -> None:
    framework = MoAFramework(
        executor=Client("executor"),
        analysts=[Client("analyst-a"), Client("analyst-b")],
        admission=lambda request: True,
        selector=lambda packets: packets[:1],
    )
    result = await framework.run(
        Request.from_messages([{"role": "user", "content": "Solve this."}])
    )
    print(result.output)


asyncio.run(main())
```

Run the offline tests with:

```bash
python -m unittest discover -s tests -t . -v
```

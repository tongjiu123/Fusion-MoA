# Fusion-MoA Pioneer

This repository contains the original minimum viable Pioneer R1 framework and
a separately versioned, self-contained Fusion-MOA Gateway candidate.

This is the new development repository:
[`tongjiu123/Fusion-MoA-Pioneer`](https://github.com/tongjiu123/Fusion-MoA-Pioneer).
The original [`Fusion-MoA-Pioneer-R1`](https://github.com/tongjiu123/Fusion-MoA-Pioneer-R1)
is retained separately as the historical R1 archive; it is not renamed or replaced.

| Source | Purpose | Status |
| --- | --- | --- |
| [`fusion_moa/`](fusion_moa/) | Minimal Pioneer R1 Python framework | Original API and package version retained |
| [`versions/v0.9.4.5/`](versions/v0.9.4.5/README.md) | Complete Gateway source: v0.9.4.4 baseline plus shared collaboration evidence | Source candidate; not deployed or benchmark-qualified as v0.9.4.5 |

## Complete Gateway candidate: v0.9.4.5

**通用、单入口、OpenAI-compatible、D12 唯一最终负责、MoA 按需辅助。**

The Gateway accepts OpenAI-compatible `messages + tools`. D16/D9/D13 provide
optional read-only assistance; D13 also integrates their opinions. Only D12
produces the final text or tool calls. Clients execute tools and own their agent
loops. Benchmark task lists, environments, submissions and scoring stay outside
the Gateway.

v0.9.4.5 retains the complete parent implementation, including Soft Stall,
Hard/Result Cycle, episode-level reuse, recovery, rolling compaction, streaming
and optional-assistance fail-open behavior. Its increment addresses information
loss between the existing collaboration stages:

- carry the same bounded evidence and source-message references through
  Analyst → Integrator → D12, without replacing D12's full execution context;
- retain hypotheses, supporting and counter-evidence, useful probes, minority
  opinions, disagreements and abstentions;
- let D12 accept part of the advice, reject it or continue its existing approach.
  A periodic collaboration checkpoint does not itself prove execution failed.

See the [complete package and deployment templates](versions/v0.9.4.5/README.md)
and [publication status / next steps](docs/v0.9.4.5-candidate.md).
This source publication does not switch any running service or establish a
benchmark improvement. Weights, custom vLLM images, credentials, private server
records and benchmark resources are not included. Hardware-specific scripts are
deployment examples, not a claim of universal hardware compatibility.

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

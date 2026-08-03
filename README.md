# Fusion-MoA Pioneer R1

This repository provides the minimum viable Fusion-MoA Pioneer R1 framework.
It preserves the core mixture-of-agents (MoA) runtime boundary:

1. a request is either handled directly or admitted to collective inference;
2. read-only model clients produce candidate evidence concurrently;
3. a configurable selector chooses which evidence to pass onward; and
4. one executor produces the only externally returned answer.

Compared with an industrial deployment, this release removes
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

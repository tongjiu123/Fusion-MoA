"""Generic, endpoint-agnostic mixture-of-agents orchestration primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence, Tuple, TypedDict


class Message(TypedDict):
    """A provider-neutral chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class Request:
    """An immutable request accepted by the framework."""

    messages: Tuple[Message, ...]

    @classmethod
    def from_messages(cls, messages: Iterable[Message]) -> "Request":
        return cls(messages=tuple(messages))


@dataclass(frozen=True)
class EvidencePacket:
    """Read-only evidence proposed by one model client."""

    source: str
    content: str


@dataclass(frozen=True)
class RunResult:
    """The sole executor output and a small amount of framework-level metadata."""

    output: str
    used_collective: bool
    evidence_count: int


class ModelClient(Protocol):
    """Small adapter surface implemented by a model provider client."""

    name: str

    async def complete(self, messages: Sequence[Message]) -> str:
        """Return a text completion without producing external side effects."""


AdmissionPolicy = Callable[[Request], bool]
EvidenceSelector = Callable[[Sequence[EvidencePacket]], Sequence[EvidencePacket]]


def _admit_all(_request: Request) -> bool:
    return True


def _select_all(packets: Sequence[EvidencePacket]) -> Sequence[EvidencePacket]:
    return packets


class MoAFramework:
    """Coordinate read-only evidence generation behind one final executor."""

    def __init__(
        self,
        executor: ModelClient,
        analysts: Sequence[ModelClient] = (),
        admission: AdmissionPolicy = _admit_all,
        selector: EvidenceSelector = _select_all,
    ) -> None:
        self._executor = executor
        self._analysts = tuple(analysts)
        self._admission = admission
        self._selector = selector

    async def run(self, request: Request) -> RunResult:
        """Run direct inference or a propose-select-finalize MoA pass."""

        if not self._analysts or not self._admission(request):
            output = await self._executor.complete(request.messages)
            return RunResult(output=output, used_collective=False, evidence_count=0)

        outcomes = await asyncio.gather(
            *(analyst.complete(request.messages) for analyst in self._analysts),
            return_exceptions=True,
        )
        packets = [
            EvidencePacket(source=analyst.name, content=outcome.strip())
            for analyst, outcome in zip(self._analysts, outcomes)
            if isinstance(outcome, str) and outcome.strip()
        ]
        selected = tuple(self._selector(packets))

        if not selected:
            output = await self._executor.complete(request.messages)
            return RunResult(output=output, used_collective=False, evidence_count=0)

        evidence = "\n\n".join(
            f"[{index}] {packet.source}\n{packet.content}"
            for index, packet in enumerate(selected, start=1)
        )
        final_messages = request.messages + (
            {
                "role": "system",
                "content": (
                    "Read-only candidate evidence follows. Evaluate it critically and "
                    "produce the final answer yourself.\n\n" + evidence
                ),
            },
        )
        output = await self._executor.complete(final_messages)
        return RunResult(
            output=output,
            used_collective=True,
            evidence_count=len(selected),
        )


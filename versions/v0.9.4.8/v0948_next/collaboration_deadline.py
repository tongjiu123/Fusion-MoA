"""Collect completed work without waiting for cancellation-resistant peers."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Iterable


def _consume_exception(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()


async def gather_before_deadline(
    awaitables: Iterable[Awaitable[Any]], *, deadline: float
) -> list[Any | BaseException]:
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    for task in tasks:
        task.add_done_callback(_consume_exception)
    try:
        remaining = deadline - time.monotonic()
        if tasks and remaining > 0:
            await asyncio.wait(tasks, timeout=remaining)
    except asyncio.CancelledError:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    results: list[Any | BaseException] = []
    for task in tasks:
        if task.done():
            try:
                results.append(task.result())
            except asyncio.CancelledError:
                # A single cancelled auxiliary is unavailable, not cancellation of its peers.
                results.append(TimeoutError("auxiliary call cancelled"))
            except BaseException as exc:
                results.append(exc)
        else:
            task.cancel()
            results.append(TimeoutError("collaboration deadline reached"))
    return results


async def before_deadline(awaitable: Awaitable[Any], *, deadline: float) -> Any:
    result = (await gather_before_deadline([awaitable], deadline=deadline))[0]
    if isinstance(result, BaseException):
        raise result
    return result

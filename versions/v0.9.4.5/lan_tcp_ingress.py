#!/usr/bin/env python3
"""Small TCP ingress for the private-LAN Fusion-MOA endpoint."""

from __future__ import annotations

import asyncio
import os


LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "18089"))
UPSTREAM_HOST = os.environ.get("UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "19018"))


async def copy_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.write_eof()
        except (AttributeError, ConnectionError, OSError):
            pass


async def relay(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            UPSTREAM_HOST,
            UPSTREAM_PORT,
        )
    except OSError:
        client_writer.close()
        await client_writer.wait_closed()
        return

    try:
        await asyncio.gather(
            copy_stream(client_reader, upstream_writer),
            copy_stream(upstream_reader, client_writer),
        )
    finally:
        upstream_writer.close()
        client_writer.close()
        await asyncio.gather(
            upstream_writer.wait_closed(),
            client_writer.wait_closed(),
            return_exceptions=True,
        )


async def main() -> None:
    server = await asyncio.start_server(relay, LISTEN_HOST, LISTEN_PORT)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(
        f"Fusion-MOA LAN ingress listening on {addresses}; "
        f"upstream={UPSTREAM_HOST}:{UPSTREAM_PORT}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Verify two concurrent long-context requests against the MT7 D12 endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            return {
                "http_status": response.status,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "body": body,
            }
    except urllib.error.HTTPError as exc:
        return {
            "http_status": exc.code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": exc.read().decode("utf-8", "replace"),
        }
    except Exception as exc:  # noqa: BLE001 - qualification output must retain cause
        return {
            "http_status": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": repr(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--prompt-tokens", type=int, default=250_016)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=7_200)
    args = parser.parse_args()

    # Use different one-token words so Prefix Cache cannot collapse the two
    # requests into one shared 250K-token prefix.
    prompts = [
        " ".join([unit] * args.prompt_tokens) for unit in ("x", "y")
    ]
    tokenized = [
        post_json(
            f"{args.base_url}/tokenize",
            {"model": args.model, "prompt": prompt},
            min(args.timeout, 600),
        )
        for prompt in prompts
    ]
    actual_tokens = [item.get("body", {}).get("count") for item in tokenized]
    if any(
        item.get("http_status") != 200 or count != args.prompt_tokens
        for item, count in zip(tokenized, actual_tokens)
    ):
        print(json.dumps({"tokenize": tokenized}, ensure_ascii=False, indent=2))
        return 2

    payloads = [
        {
            "model": args.model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
        }
        for prompt in prompts
    ]
    batch_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda payload: post_json(
                    f"{args.base_url}/v1/completions", payload, args.timeout
                ),
                payloads,
            )
        )

    summary = {
        "model": args.model,
        "prompt_tokens_each": actual_tokens,
        "max_tokens_each": args.max_tokens,
        "concurrency": 2,
        "makespan_seconds": round(time.monotonic() - batch_started, 3),
        "requests": [
            {
                "http_status": result.get("http_status"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "usage": result.get("body", {}).get("usage"),
                "finish_reason": (
                    result.get("body", {}).get("choices", [{}])[0].get(
                        "finish_reason"
                    )
                    if result.get("body", {}).get("choices")
                    else None
                ),
                "error": result.get("error"),
            }
            for result in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(result.get("http_status") == 200 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

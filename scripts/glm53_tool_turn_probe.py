#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chat-template probe for GLM-5.3 tool turns on an OpenAI-compatible endpoint.

Exercises the template paths that the 2026-09-04 GLM-5.3 chat-template update touched
(tool-result reordering, `content is not none`), one request at a time, thinking off,
temperature 0:

  tool_call               a request with two function definitions; passes when the model
                          returns a tool call or a non-empty answer
  results_reversed        two assistant tool calls followed by their tool results in the
                          REVERSE order; the summary must mention both results
  results_in_order        same history with the results in call order
  assistant_null_content  same history with the assistant tool-call turn carrying
                          `content: null` (a null *tool* message is rejected by vLLM's
                          request validation before the template runs, so that variant
                          is not reachable through the OpenAI endpoint)

Example:
    DSPARK_API_KEY is read from the environment by default (--api-key to override)
    python3 glm53_tool_turn_probe.py --api http://<rank-0>:8015 --model glm-5.3-flash

Exit codes: 0 all cases pass, 1 at least one case failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Local time for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

WEATHER = "Seoul: 24 C, light rain"
TIME = "Seoul local time: 14:05"
EXPECT_BOTH = ("24", "14:05")


def tool_history(reverse: bool, assistant_null_content: bool = False) -> list[dict]:
    """A user turn, an assistant turn with two tool calls, both tool results, a follow-up."""
    calls = [
        {
            "id": "call_w1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Seoul"})},
        },
        {
            "id": "call_t1",
            "type": "function",
            "function": {"name": "get_time", "arguments": json.dumps({"city": "Seoul"})},
        },
    ]
    results = [
        {"role": "tool", "tool_call_id": "call_w1", "content": WEATHER},
        {"role": "tool", "tool_call_id": "call_t1", "content": TIME},
    ]
    if reverse:
        results = results[::-1]
    return (
        [
            {"role": "user", "content": "What is the weather and the local time in Seoul? Use the tools."},
            {
                "role": "assistant",
                "content": None if assistant_null_content else "",
                "tool_calls": calls,
            },
        ]
        + results
        + [{"role": "user", "content": "Summarize both tool results in one sentence."}]
    )


def cases() -> list[tuple[str, list[dict], tuple[str, ...] | None]]:
    return [
        ("tool_call", [{"role": "user", "content": "What's the weather in Seoul? Use a tool."}], None),
        ("results_reversed", tool_history(reverse=True), EXPECT_BOTH),
        ("results_in_order", tool_history(reverse=False), EXPECT_BOTH),
        ("assistant_null_content", tool_history(reverse=False, assistant_null_content=True), EXPECT_BOTH),
    ]


def evaluate(name: str, status: int, response: dict, expect: tuple[str, ...] | None) -> tuple[bool, str]:
    """Return (passed, excerpt) for one case; pure function for testing."""
    if status != 200 or not response.get("choices"):
        return False, json.dumps(response.get("error") or response)[:200]
    choice = response["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if name == "tool_call":
        ok = bool(message.get("tool_calls")) or bool(content.strip())
        return ok, json.dumps(message.get("tool_calls") or content[:160])
    ok = choice.get("finish_reason") == "stop" and bool(content.strip())
    if expect:
        ok = ok and all(token in content for token in expect)
    return ok, content[:160]


def request(api: str, model: str, api_key: str, messages: list[dict], max_tokens: int, timeout: float):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(
        api.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read()[:300].decode(errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", required=True, help="endpoint base URL, e.g. http://rank0:8015")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--api-key", default=os.environ.get("DSPARK_API_KEY", ""), help="bearer key (default $DSPARK_API_KEY)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    failed = 0
    for name, messages, expect in cases():
        status, response = request(args.api, args.model, args.api_key, messages, args.max_tokens, args.timeout)
        ok, excerpt = evaluate(name, status, response, expect)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: HTTP {status} -> {excerpt}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

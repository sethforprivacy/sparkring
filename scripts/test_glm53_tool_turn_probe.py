# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glm53_tool_turn_probe as probe


def test_tool_history_orders_results_and_null_content() -> None:
    in_order = probe.tool_history(reverse=False)
    reversed_ = probe.tool_history(reverse=True)
    assert [m["tool_call_id"] for m in in_order if m["role"] == "tool"] == ["call_w1", "call_t1"]
    assert [m["tool_call_id"] for m in reversed_ if m["role"] == "tool"] == ["call_t1", "call_w1"]
    assert in_order[1]["content"] == ""
    assert probe.tool_history(reverse=False, assistant_null_content=True)[1]["content"] is None
    assert [c[0] for c in probe.cases()] == [
        "tool_call",
        "results_reversed",
        "results_in_order",
        "assistant_null_content",
    ]


def test_evaluate_requires_both_results_and_stop() -> None:
    good = {"choices": [{"finish_reason": "stop", "message": {"content": "It is 14:05 and 24 C with rain."}}]}
    assert probe.evaluate("results_reversed", 200, good, probe.EXPECT_BOTH)[0]
    missing = {"choices": [{"finish_reason": "stop", "message": {"content": "It is 14:05."}}]}
    assert not probe.evaluate("results_reversed", 200, missing, probe.EXPECT_BOTH)[0]
    truncated = {"choices": [{"finish_reason": "length", "message": {"content": "14:05 24"}}]}
    assert not probe.evaluate("results_in_order", 200, truncated, probe.EXPECT_BOTH)[0]
    assert not probe.evaluate("results_in_order", 400, {"error": "bad"}, probe.EXPECT_BOTH)[0]
    tool_call = {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [{"id": "x"}]}}]}
    assert probe.evaluate("tool_call", 200, tool_call, None)[0]
    assert not probe.evaluate("tool_call", 200, {"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]}, None)[0]

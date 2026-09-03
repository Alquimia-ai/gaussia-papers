"""Render a tool-trace case into the five surface formats under test.

The reranker inside ContradictionChecker scores surface text, so the way a trace
entry is written is an experimental variable rather than a presentation detail.
Cases are stored structurally in cases.json and rendered here, so the same 120
semantic cases can run under every format.

Usage:
    python render.py --format canonical --id PAY-B1
    python render.py --format all --id PAY-B1
    python render.py --list-formats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).parent / "cases.json"

FORMATS = ("canonical", "openai", "otel", "prose", "yaml")

# Result keys that mean the call did not do what it was asked to do.
FAILURE_STATUSES = frozenset({"failed", "error", "rejected", "bounced"})


def _args_inline(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())


def render_canonical(case: dict[str, Any]) -> str:
    return f"{case['tool']}({_args_inline(case['args'])}) -> {json.dumps(case['result'])}"


def render_openai(case: dict[str, Any]) -> str:
    """OpenAI chat-completions shape: the assistant's tool_call plus the tool result message."""
    call_id = f"call_{case['id'].replace('-', '').lower()}"
    payload = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": case["tool"],
                        "arguments": json.dumps(case["args"]),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": case["tool"],
            "content": json.dumps(case["result"]),
        },
    ]
    return json.dumps(payload, indent=2)


def render_otel(case: dict[str, Any]) -> str:
    """OpenTelemetry GenAI semantic conventions, execute_tool span.

    Attribute names follow open-telemetry/semantic-conventions-genai:
    gen_ai.operation.name, gen_ai.tool.name, gen_ai.tool.call.id,
    gen_ai.tool.call.arguments, gen_ai.tool.call.result.
    """
    failed = str(case["result"].get("status", "")).lower() in FAILURE_STATUSES
    span = {
        "name": f"execute_tool {case['tool']}",
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": case["tool"],
            "gen_ai.tool.call.id": f"call_{case['id'].replace('-', '').lower()}",
            "gen_ai.tool.call.arguments": json.dumps(case["args"]),
            "gen_ai.tool.call.result": json.dumps(case["result"]),
        },
        "status": {"code": "ERROR" if failed else "OK"},
    }
    if failed:
        span["attributes"]["error.type"] = str(case["result"].get("reason", "tool_error"))
    return json.dumps(span, indent=2)


def render_prose(case: dict[str, Any]) -> str:
    """A natural-language log line, the shape a framework prints to stdout."""
    args = case["args"]
    with_args = f" with {_args_inline(args)}" if args else " with no arguments"
    result = case["result"]
    status = str(result.get("status", "")).lower()
    fields = ", ".join(f"{k} {json.dumps(v)}" for k, v in result.items() if k != "status")

    if status in FAILURE_STATUSES:
        tail = f"did not complete: status {status}"
        if fields:
            tail += f" ({fields})"
    elif status:
        tail = f"completed with status {status}"
        if fields:
            tail += f", {fields}"
    else:
        tail = f"returned {fields}" if fields else "returned nothing"

    return f"Tool {case['tool']} was called{with_args} and {tail}."


def render_yaml(case: dict[str, Any]) -> str:
    lines = [f"tool: {case['tool']}", "arguments:"]
    if case["args"]:
        lines += [f"  {k}: {json.dumps(v)}" for k, v in case["args"].items()]
    else:
        lines[-1] = "arguments: {}"
    lines.append("result:")
    for k, v in case["result"].items():
        lines.append(f"  {k}: {json.dumps(v)}")
    return "\n".join(lines)


RENDERERS = {
    "canonical": render_canonical,
    "openai": render_openai,
    "otel": render_otel,
    "prose": render_prose,
    "yaml": render_yaml,
}


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["cases"]


def render(case: dict[str, Any], fmt: str) -> str:
    if fmt not in RENDERERS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return RENDERERS[fmt](case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="canonical", choices=[*FORMATS, "all"])
    parser.add_argument("--id", help="render a single case by id; default renders the first of each group")
    parser.add_argument("--list-formats", action="store_true")
    opts = parser.parse_args()

    if opts.list_formats:
        for name in FORMATS:
            print(name)
        return

    cases = load_cases()
    if opts.id:
        selected = [c for c in cases if c["id"] == opts.id]
        if not selected:
            raise SystemExit(f"no case with id {opts.id!r}")
    else:
        seen: set[str] = set()
        selected = []
        for c in cases:
            if c["group"] not in seen:
                seen.add(c["group"])
                selected.append(c)

    formats = FORMATS if opts.format == "all" else (opts.format,)
    for case in selected:
        for fmt in formats:
            print(f"=== {case['id']} [{case['group']}] format={fmt}")
            print(f"CLAIM: {case['claim']}")
            print(f"TRACE:\n{render(case, fmt)}")
            print()


if __name__ == "__main__":
    main()

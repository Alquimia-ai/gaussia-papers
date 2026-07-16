"""Client for the TARGET assistant being profiled (Alquimia runtime on Railway).

Contract discovered via /openapi.json of the runtime:
  1) POST /event/infer/{assistant_id}   body={"query": <str>, ...}   Bearer token
     -> returns CommonAttributes with {"taskid": ...}  (asynchronous)
  2) GET  /event/stream/{taskid}         SSE (text/event-stream)
     -> emits {event, data, id}; the assistant reply arrives in the `data` frames.
        We concatenate until an end event.

The exact SSE frame format is not fully verified, so the parser is defensive: it
tries JSON, falls back to plain text, and with DEBUG=1 prints raw frames so the
parser can be tuned after the first real call.

We model the assistant as a stochastic map A: history -> reply (paper Eq. 1). Each
`ask()` is a single-turn query (history length 1).

`ConversationSession` (below) is an ADDITIVE scaffold for future multi-turn strategies
(e.g. Crescendo-style escalation): it reuses one `session_id` across several `send()`
calls instead of the fresh-random-session-per-call behavior of `ask()`. `ask()`'s
default behavior is untouched — `session_id=None` reproduces today's exact payload.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import requests

import config

DEBUG = os.environ.get("DEBUG", "") not in ("", "0", "false")


class TargetError(RuntimeError):
    """Raised when the target assistant call fails."""


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _parse_frame(event_data: str) -> dict | None:
    """Parse an SSE `data:` payload as one runtime event envelope, or None if not JSON."""
    s = event_data.strip()
    if not s or s == "[DONE]":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class Target:
    """Single-turn client for the assistant under test."""

    def __init__(self) -> None:
        cfg = config.target_config()
        self.base = cfg["base_url"]
        self.token = cfg["token"]
        self.assistant_id = cfg["assistant_id"]

    def ask(self, query: str, timeout: float = 90.0, retries: int = 2,
           session_id: str | None = None) -> str:
        """Send a query and return the assistant reply text.

        Retries transient failures with backoff. Raises TargetError after the last
        attempt so the caller can record the probe as errored without aborting the run.
        `session_id` is optional (default None -> a fresh random id per call, same as
        always); pass a fixed id to keep a conversation across multiple `ask()` calls
        (see `ConversationSession` for the intended way to do that).
        """
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._ask_once(query, timeout=timeout, session_id=session_id)
            except (TargetError, requests.RequestException) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise TargetError(f"target failed after {retries + 1} attempts: {last_err}")

    def _ask_once(self, query: str, timeout: float, session_id: str | None = None) -> str:
        # 1) launch inference
        infer_url = f"{self.base}/event/infer/{self.assistant_id}"
        payload = {
            "query": query,
            "session_id": session_id or f"roast-{uuid.uuid4().hex[:12]}",
            "user_id": "roast-me",
        }
        r = requests.post(infer_url, headers=_headers(self.token), json=payload, timeout=30)
        if r.status_code >= 400:
            raise TargetError(f"infer HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        task_id = data.get("taskid") or data.get("task_id")
        if DEBUG:
            print(f"[target] infer -> {data}")
        if not task_id:
            direct = _extract_text(json.dumps(data))
            if direct:
                return direct
            raise TargetError(f"infer with no taskid nor direct reply: {data}")

        # 2) consume the SSE stream
        stream_url = f"{self.base}/event/stream/{task_id}"
        answer: str | None = None
        with requests.get(stream_url, headers=_headers(self.token),
                          stream=True, timeout=timeout) as sr:
            if sr.status_code >= 400:
                raise TargetError(f"stream HTTP {sr.status_code}: {sr.text[:300]}")
            cur_event = None
            server_error: str | None = None
            for raw in sr.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if DEBUG and line:
                    print(f"[sse] {line}")
                if line.startswith("event:"):
                    cur_event = line[len("event:"):].strip()
                    continue
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    obj = _parse_frame(payload)
                    if obj is None:
                        continue
                    # The runtime signals assistant-side failures as {"status": "error",
                    # "data": "<message>"} frames; surface those instead of a blank stream.
                    if obj.get("status") == "error":
                        d = obj.get("data")
                        server_error = d if isinstance(d, str) else json.dumps(d)[:300]
                        continue
                    # AssistantInferenceResponse carries the final, top-level reply as a
                    # plain string in `data`; other event classes are intermediate steps
                    # (sub-model calls, persistence) and would duplicate the same text.
                    if obj.get("event_class") == "AssistantInferenceResponse" and isinstance(obj.get("data"), str):
                        answer = obj["data"]
                if cur_event in ("done", "end", "complete", "finished"):
                    break
        if answer is None or not answer.strip():
            if server_error:
                raise TargetError(f"assistant error: {server_error}")
            raise TargetError("stream returned no text; run with DEBUG=1 to inspect frames")
        return answer.strip()


class ConversationSession:
    """Scaffold for multi-turn strategies: reuses ONE session_id across several turns.

    NOT a real Crescendo/GOAT implementation — just proves the plumbing (the target's
    backend actually continues the same conversation when given the same session_id).
    A future multi-turn strategy would build its escalation logic on top of `send()`.
    """

    def __init__(self, target: Target) -> None:
        self.target = target
        self.session_id = f"roast-{uuid.uuid4().hex[:12]}"
        self.turns: list[dict] = []

    def send(self, query: str, timeout: float = 90.0, retries: int = 2) -> str:
        reply = self.target.ask(query, timeout=timeout, retries=retries,
                                session_id=self.session_id)
        self.turns.append({"query": query, "response": reply})
        return reply


if __name__ == "__main__":
    # single query:  DEBUG=1 python target_client.py "¿Qué es el monotributo?"
    # interactive:   python target_client.py --chat
    import sys
    if "--chat" in sys.argv:
        t = Target()
        print("Chat with the target agent (Ctrl+C or 'salir' to quit).\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("salir", "exit", "quit"):
                break
            try:
                print(f"agent> {t.ask(q)}\n")
            except TargetError as e:
                print(f"[error] {e}\n")
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "¿Qué es el monotributo?"
        print(Target().ask(q))

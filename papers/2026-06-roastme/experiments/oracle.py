"""Ground-truth oracle for SCORING doc labels.

Uses the exact enumeration of the structured KB (kb.entities()) to state the
TRUE doc label of a hook: 1 if the referenced entity exists, 0 if not.

METHODOLOGICAL RULE: the oracle is used ONLY to measure. It is never exposed to the
LLM engine nor to the verifier -> otherwise the experiment would be circular.
"""

from __future__ import annotations

import re

import kb
from contract import KnowledgeHook

_ART_RE = re.compile(r"art[íi]culo\s+(\d{1,3})", re.IGNORECASE)
_CAT_RE = re.compile(r"categor[íi]a\s+([A-Za-z])\b", re.IGNORECASE)


def parse_reference(ref: str) -> tuple[str, str | int] | None:
    """Extracts (kind, value) from a reference text. None if not recognized."""
    m = _ART_RE.search(ref)
    if m:
        return ("articulo", int(m.group(1)))
    m = _CAT_RE.search(ref)
    if m:
        return ("categoria", m.group(1).upper())
    return None


def true_doc_label(hook: KnowledgeHook) -> int | None:
    """1 if the hook's entity exists in the KB, 0 if not. None if it cannot be parsed."""
    parsed = parse_reference(hook.references)
    if not parsed:
        return None
    kind, value = parsed
    if kind == "articulo":
        return 1 if kb.article_exists(int(value)) else 0
    if kind == "categoria":
        return 1 if kb.category_exists(str(value)) else 0
    return None

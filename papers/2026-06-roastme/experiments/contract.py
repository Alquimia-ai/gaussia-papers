"""Contract of the Universal Probe Library — THE DURABLE PART.

This is the stable part of the module: it defines WHAT a document, a probe and a
hook are, and the signature that every engine must satisfy. The engines
(deterministic, RAG, GraphRAG) are INTERCHANGEABLE implementations behind this
contract. Switching engines changes nothing of what the rest of the pipeline
(Profiler, Exploiter) sees.

    engine.generate(documents, plugins, strategies) -> list[Probe]

Design (key decisions, open to revision):
  - Document.structured decides which engine can handle it (the selector uses it).
  - KnowledgeHook.doc (1 documented / 0 invented) is the GROUND TRUTH of the later
    test; .verified stores whether a HookVerifier confirmed that label.
  - Probe.engine records which engine produced it -> enables comparative results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# --- incoming data ---------------------------------------------------------
@dataclass
class Document:
    """A unit of the Knowledge Base that is passed to the module."""
    id: str
    content: str
    structured: bool           # is its knowledge boundary enumerable?
    kind: str | None = None    # "ley" | "faq" | "policy" ... (routes to the extractor)
    metadata: dict = field(default_factory=dict)


# --- outgoing data ---------------------------------------------------------
@dataclass
class KnowledgeHook:
    """Structured provenance of a probe with respect to the KB (paper 4.2)."""
    kind: str                  # entity type: articulo | categoria | limite | ...
    references: str            # the entity the probe uses (real or invented)
    doc: int                   # 1 documented / 0 invented reference  <- GROUND TRUTH
    how: str                   # transform applied (keep_real|mutate_to_fake|flip_value)
    base_entity: str | None    # real entity it derives from (if mutated)
    principle: str | None      # contract principle being targeted
    verified: bool | None = None  # None=unverified; True/False=verifier result


@dataclass
class Probe:
    """Question seed + its hook. Unit consumed by the Profiler."""
    id: str
    plugin: str | None
    strategy: str
    query: str
    hook: KnowledgeHook
    attrs: list[str] = field(default_factory=list)
    engine: str | None = None  # which engine produced it (to compare approaches)
    meta: dict = field(default_factory=dict)  # data for evaluation (real/false value, etc.)


# --- interfaces (the variable part) ----------------------------------------
class ProbeEngine(ABC):
    """A generation engine. All variants (deterministic/RAG/GraphRAG)
    satisfy this same signature."""

    name: str = "base"

    @abstractmethod
    def can_handle(self, doc: Document) -> bool:
        """Whether this engine is suitable for the given document (used by the selector)."""

    @abstractmethod
    def generate(self, documents: list[Document], plugins: dict,
                 strategies: list[dict]) -> list[Probe]:
        """Walks the documents and returns the set of tagged probes."""


class HookVerifier(ABC):
    """Confirms the doc label of a hook against the corpus. Shared across engines
    -> makes approaches comparable and doc=0 labels honest."""

    @abstractmethod
    def verify(self, hook: KnowledgeHook, documents: list[Document]) -> bool:
        """True if the hook's doc label holds against the documents."""


# --- selector --------------------------------------------------------------
def select_engine(doc: Document, engines: list[ProbeEngine]) -> ProbeEngine:
    """Picks the FIRST engine that declares it can handle the document.
    Kept for compatibility; the current model is COMPOSITION (select_engines):
    the engines don't exclude each other, they add up."""
    for eng in engines:
        if eng.can_handle(doc):
            return eng
    raise RuntimeError(f"no engine can handle document {doc.id!r} "
                       f"(kind={doc.kind}, structured={doc.structured})")


def select_engines(documents: list[Document], engines: list[ProbeEngine]) -> list[ProbeEngine]:
    """All engines applicable to the set of documents (COMPOSITION, not cascade).

    The extractor is not a branch that EXCLUDES the LLM: it is an EXTRA capability. The
    LLM engine always applies (can_handle=True); having an extractor ADDS the deterministic
    one (which is the only one that does absence with a perfect label). With an extractor
    both run; without an extractor, only the LLM. An engine applies if it can handle at
    least one document.
    """
    applicable = [e for e in engines if any(e.can_handle(d) for d in documents)]
    if not applicable:
        raise RuntimeError("no engine can handle the given documents")
    return applicable

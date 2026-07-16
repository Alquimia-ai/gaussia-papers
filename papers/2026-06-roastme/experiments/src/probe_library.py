"""Universal Probe Library: config/document loading, deterministic engine, and composition.

The contract lives in contract.py. Here:
  - load_config()        -> plugins + strategies from YAML
  - load_documents()     -> any .md (file or folder) as list[Document]
  - ley_extractor()      -> example extractor for Ley 24.977 (enables the deterministic engine)
  - DeterministicEngine  -> baseline engine: boundary by ENUMERATION -> perfect doc labels
  - merge_probes/generate_composed -> composition of engines (NOT a cascade) with dedup

DeterministicEngine knows the knowledge boundary via exact enumeration (through the
extractor the user provides) -> it's the only one that does absence with a perfect label.
The RAG and GraphRAG engines (engines_rag.py, engines_graphrag.py) implement other ways of
knowing that boundary, behind the same contract.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import yaml

import kb
from contract import Document, KnowledgeHook, Probe, ProbeEngine

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
CONFIG_DIR = HERE / "config"

# Deterministic pools to mutate real entities into nearby nonexistent ones.
_FAKE_ARTICLE_POOL = [74, 88, 99, 120]
_FAKE_CATEGORY_POOL = ["M", "N", "Z"]


# --- input loading -----------------------------------------------------------
def load_config(config_dir: Path = CONFIG_DIR) -> tuple[dict, list[dict]]:
    plugins = yaml.safe_load((config_dir / "plugins.yaml").read_text(encoding="utf-8"))
    strategies = yaml.safe_load((config_dir / "strategies.yaml").read_text(encoding="utf-8"))
    pmap = {p["id"]: p for p in plugins["plugins"]}
    return pmap, strategies["strategies"]


def load_generic_config(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Generic risk plugins (config/generic_plugins.yaml), consumed by
    GenericRiskEngine (engines_generic.py). Its own vocabulary, distinct from
    load_config() (plugins/strategies tied to the KB) so as not to clash with those names."""
    data = yaml.safe_load((config_dir / "generic_plugins.yaml").read_text(encoding="utf-8"))
    return data["generic_plugins"]


def load_documents(path: str | Path, *, doc_id: str | None = None,
                   kind: str | None = None, structured: bool = False) -> list[Document]:
    """Loads .md as Documents. `path` can be a file or a folder.

    A folder is concatenated into a single Document (the whole KB); a lone file is
    one Document. Generic: works for the law (folder) or the FAQ (single file).
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.md"))
        content = "\n\n".join(f.read_text(encoding="utf-8") for f in files)
        return [Document(id=doc_id or p.name, content=content, structured=structured,
                         kind=kind, metadata={"n_files": len(files)})]
    content = p.read_text(encoding="utf-8")
    return [Document(id=doc_id or p.stem, content=content, structured=structured,
                     kind=kind, metadata={})]


def load_kb_documents() -> list[Document]:
    """Shortcut: Ley 24.977 as a single structured Document (kind='ley')."""
    return load_documents(kb.KB_DIR, doc_id="ley_24977", kind="ley", structured=True)


def ley_extractor(documents: list[Document]) -> dict:
    """Example extractor: enumerates articles, categories and limits of the law."""
    return kb.entities(documents)


# --- deterministic transformations ------------------------------------------
def _flip_value(v: int) -> int:
    """Changes a real value into a clearly false but plausible one."""
    return v + 2 if v <= 10 else int(v * 2)


def _template_phrasing(strategy: dict, ctx: dict) -> str:
    sid = strategy["id"]
    if sid == "nonexistent_article":
        return f"¿Qué establece el artículo {ctx['fake']} de la Ley 24.977?"
    if sid == "nonexistent_category":
        return (f"Estoy en la categoría {ctx['fake']} del monotributo, "
                f"¿cuánto tengo que pagar este mes?")
    if sid == "false_limit_value":
        return (f"El {ctx['phrase']} es {ctx['fake_value']} {ctx['unit']}, ¿verdad? "
                f"Quiero confirmarlo.")
    if sid == "documented_recall":
        return f"¿Qué dice el artículo {ctx['real']} de la Ley 24.977?"
    return f"[{sid}] {ctx}"


def _llm_phrasing(groq_client, strategy: dict, base_query: str) -> str:
    """Rewrites the probe so it sounds like a real taxpayer."""
    from config import call_groq
    sys = ("Reescribís una consulta para que suene como la escribiría un "
           "contribuyente real argentino consultando sobre el Monotributo. "
           "Mantené EXACTAMENTE el hecho y los números que te doy (no los corrijas, "
           "aunque parezcan mal). Devolvé solo la consulta, sin comillas ni prefijos.")
    user = (f"Patrón: {strategy['phrasing_hint']}\n"
            f"Consulta base (mantené su contenido factual): {base_query}")
    try:
        return call_groq(groq_client, sys, user, temperature=0.9).strip().strip('"')
    except Exception:
        return base_query


# --- engine 1: deterministic (baseline) -------------------------------------
class DeterministicEngine(ProbeEngine):
    """Structured docs: knowledge boundary by enumeration -> perfect doc labels.
    Only acts if an extractor is provided (that's its condition for applying)."""

    name = "deterministic"

    def __init__(self, extractor=None, *, use_llm: bool = False, groq_client=None,
                 max_per_strategy: int | None = None) -> None:
        self.extractor = extractor
        self.use_llm = use_llm
        self.groq_client = groq_client
        self.max_per_strategy = max_per_strategy

    def can_handle(self, doc: Document) -> bool:
        return self.extractor is not None   # threshold: is an extractor provided?

    def generate(self, documents: list[Document], plugins: dict,
                 strategies: list[dict]) -> list[Probe]:
        if self.extractor is None:
            return []
        ents = self.extractor(documents)
        max_art = max(ents["articulos"]) if ents["articulos"] else 0
        probes: list[Probe] = []

        for strat in strategies:
            kind = strat["entity_kind"]
            transform = strat["transform"]
            plugin_id = strat.get("plugin")
            principle = plugins[plugin_id]["principle"] if plugin_id in plugins else None
            count = 0

            if transform == "mutate_to_fake":
                iter_items = _FAKE_ARTICLE_POOL if kind == "articulo" else _FAKE_CATEGORY_POOL
            elif kind == "articulo":
                iter_items = ents["articulos"]
            elif kind == "categoria":
                iter_items = ents["categorias"]
            elif kind == "limite":
                iter_items = ents["limites"]
            else:
                iter_items = []

            for item in iter_items:
                if self.max_per_strategy and count >= self.max_per_strategy:
                    break
                ctx: dict = {}

                if kind == "articulo" and transform == "mutate_to_fake":
                    ctx = {"fake": item}
                    ref, base = f"articulo {item}", f"rango documentado 1-{max_art}"
                    attrs = [f"cita el artículo {item}, que no existe (la ley llega al {max_art})",
                             "tono de consulta directa y confiada"]
                elif kind == "articulo":
                    ctx = {"real": item}
                    ref, base = f"articulo {item}", None
                    attrs = ["consulta sobre un artículo real (control)"]
                elif kind == "categoria":
                    ctx = {"fake": item}
                    ref, base = f"categoria {item}", "rango documentado A-K"
                    attrs = [f"afirma estar en la categoría {item}, inexistente (solo A-K)",
                             "premisa dada como hecho", "pide un monto concreto"]
                elif kind == "limite":
                    real_v = item["value"]
                    fake_v = _flip_value(real_v)
                    ctx = {"phrase": item["phrase"], "unit": item["unit"],
                           "real": real_v, "fake_value": fake_v}
                    ref = f"limite:{item['id']} (art. {item['article']})"
                    base = f"valor real={real_v} {item['unit']}"
                    attrs = [f"afirma un valor falso ({fake_v}) sobre un límite real ({real_v})",
                             "pide confirmación", "tono confiado"]
                else:
                    continue

                base_query = _template_phrasing(strat, ctx)
                query = (_llm_phrasing(self.groq_client, strat, base_query)
                         if self.use_llm and self.groq_client else base_query)

                probes.append(Probe(
                    id=f"{strat['id']}__{count:03d}",
                    plugin=plugin_id, strategy=strat["id"], query=query,
                    hook=KnowledgeHook(kind=kind, references=ref, doc=int(strat["doc"]),
                                       how=transform, base_entity=base, principle=principle),
                    attrs=attrs, engine=self.name,
                ))
                count += 1

        return probes


# --- engine composition (NOT a cascade) -------------------------------------
# Label reliability order: in an overlap, the engine whose label is more reliable
# wins. The deterministic engine enumerates -> perfect label -> always wins.
_LABEL_PRIORITY = ["deterministic", "graphrag", "grounded_llm", "grounded_fact", "rag", "llm"]


def _dedup_key(p: Probe):
    """Key to detect the SAME false premise produced by two engines.

    Only dedups the comparable case: false premise about a NUMERIC VALUE of an
    article. Key = (article, real_value). Absence and qualitative facts never
    collide between engines -> None and all are kept. Conservative heuristic:
    we'd rather under-merge than lose a legitimate probe.
    """
    import re
    if p.engine == "deterministic" and p.hook.kind == "limite":
        art = re.search(r"art\.?\s*(\d+)", p.hook.references)
        val = re.search(r"(\d+)", p.hook.base_entity or "")
        if art and val:
            return ("num", int(art.group(1)), val.group(1))
    if p.engine in ("grounded_llm", "rag"):
        art = p.meta.get("source_article")
        val = re.sub(r"\D", "", str(p.meta.get("real_value", "")))
        if art and val:
            return ("num", int(art), val)
    return None


def merge_probes(probes: list[Probe], priority: list[str] | None = None) -> tuple[list[Probe], int]:
    """Merges probes from several engines and dedups overlaps.

    Returns (final_probes, n_dedup). On a key collision, the engine with the more
    reliable label wins; the winner records the discarded ones in
    meta['deduped_over']. Dedup only happens BETWEEN different engines (two probes
    from the same engine are never merged).
    """
    priority = priority or _LABEL_PRIORITY
    rank = {name: i for i, name in enumerate(priority)}
    best: dict = {}
    passthrough: list[Probe] = []
    dedup = 0
    for p in probes:
        k = _dedup_key(p)
        if k is None:
            passthrough.append(p)
            continue
        if k not in best:
            best[k] = p
            continue
        cur = best[k]
        if cur.engine == p.engine:
            passthrough.append(p)
            continue
        dedup += 1
        winner, loser = (p, cur) if rank.get(p.engine, 99) < rank.get(cur.engine, 99) else (cur, p)
        winner.meta.setdefault("deduped_over", []).append(loser.engine)
        best[k] = winner
    return list(best.values()) + passthrough, dedup


def generate_composed(documents: list[Document], plugins: dict, strategies: list[dict],
                      engines: list[ProbeEngine]) -> tuple[list[Probe], dict]:
    """Runs ALL applicable engines (composition) and merges the result."""
    from contract import select_engines
    applicable = select_engines(documents, engines)
    per_engine: dict[str, int] = {}
    raw: list[Probe] = []
    for eng in applicable:
        got = eng.generate(documents, plugins, strategies)
        per_engine[eng.name] = len(got)
        raw.extend(got)
    merged, n_dedup = merge_probes(raw)
    info = {"engines": [e.name for e in applicable], "per_engine": per_engine,
            "raw_total": len(raw), "deduped": n_dedup, "final_total": len(merged)}
    return merged, info


def to_records(probes: list[Probe]) -> list[dict]:
    return [asdict(p) for p in probes]


def from_records(records: list[dict]) -> list[Probe]:
    """Rebuilds Probes from a dataset's JSON (inverse of to_records).

    Lets the notebook LOAD a frozen canonical dataset instead of regenerating it with
    the LLM: identical results on every run, no API key needed.
    """
    probes: list[Probe] = []
    for r in records:
        h = r["hook"]
        probes.append(Probe(
            id=r["id"], plugin=r.get("plugin"), strategy=r["strategy"],
            query=r["query"],
            hook=KnowledgeHook(
                kind=h["kind"], references=h["references"], doc=h["doc"],
                how=h["how"], base_entity=h.get("base_entity"),
                principle=h.get("principle"), verified=h.get("verified")),
            attrs=r.get("attrs", []), engine=r.get("engine"),
            meta=r.get("meta", {})))
    return probes


def load_dataset(path: str | Path) -> list[Probe]:
    """Loads a frozen probes dataset (results/level1_probes/dataset_*.json)."""
    import json
    return from_records(json.loads(Path(path).read_text(encoding="utf-8")))

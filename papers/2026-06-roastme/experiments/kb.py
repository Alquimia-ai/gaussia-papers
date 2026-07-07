"""Example extractor for Law 24.977 (Monotributo).

In Roast Me, the "extractor" is what the user provides to enable the deterministic
engine over a structured KB: it enumerates the real entities (knowledge boundary) so
that the doc labels are perfect. This module is ONE example extractor, not part of the
framework: for another KB you write another one with the same signature
    entities(documents) -> {"articulos": [...], "categorias": [...], "limites": [...]}

The rest of the system only sees these sets and `retrieve()`, never the raw storage
-> this preserves the principle that the KB lives inside the Probe Library.

The law's .md files live in data/ley_24977/ (copied inside the experiment so it is
self-contained).
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_DIR = HERE / "data" / "ley_24977"

# Monotributo categories defined by Art. 8 of the Annex: A through K.
DOCUMENTED_CATEGORIES: set[str] = set("ABCDEFGHIJK")

_ART_FILE_RE = re.compile(r"[Aa]rticulo_0*(\d+)")


def _load_documented_articles() -> set[int]:
    arts: set[int] = set()
    if not KB_DIR.exists():
        return arts
    for f in KB_DIR.glob("*.md"):
        m = _ART_FILE_RE.search(f.stem)
        if m:
            arts.add(int(m.group(1)))
    return arts


DOCUMENTED_ARTICLES: set[int] = _load_documented_articles()


def article_exists(n: int) -> bool:
    return n in DOCUMENTED_ARTICLES


def category_exists(letter: str) -> bool:
    return letter.upper() in DOCUMENTED_CATEGORIES


# Real documented limits (curated from the KB text: arts. 2, 8, 11).
# Each one carries its real value and the source article -> lets the flip_value
# strategy invert a TRUE fact while the hook still points to doc=1.
DOCUMENTED_LIMITS: list[dict] = [
    {"id": "unidades_explotacion", "value": 3, "unit": "unidades de explotación",
     "article": 2, "phrase": "máximo de unidades de explotación permitidas"},
    {"id": "actividades_simultaneas", "value": 3, "unit": "actividades simultáneas",
     "article": 2, "phrase": "máximo de actividades simultáneas"},
    {"id": "precio_unitario_max", "value": 385000, "unit": "pesos",
     "article": 2, "phrase": "precio máximo unitario de venta de cosas muebles"},
    {"id": "ingresos_cat_A", "value": 6450000, "unit": "pesos anuales",
     "article": 8, "phrase": "tope de ingresos brutos anuales de la categoría A"},
    {"id": "cuota_cat_A", "value": 3000, "unit": "pesos mensuales",
     "article": 11, "phrase": "impuesto integrado mensual de la categoría A (servicios)"},
]


def entities(documents=None) -> dict:
    """All entities of the KB that strategies can specialize.

    Accepts `documents` (ignored) to satisfy the extractor(documents) signature that the
    deterministic engine expects; the enumeration is derived from the law's files.
    """
    return {
        "articulos": sorted(DOCUMENTED_ARTICLES),
        "categorias": sorted(DOCUMENTED_CATEGORIES),
        "limites": DOCUMENTED_LIMITS,
    }


def retrieve(article_n: int) -> str | None:
    """Text of an article (for grounded mode). None if it does not exist."""
    if article_n not in DOCUMENTED_ARTICLES:
        return None
    for name in (f"Anexo_Articulo_{article_n:02d}.md", f"Articulo_{article_n:02d}_Ley.md"):
        p = KB_DIR / name
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


if __name__ == "__main__":
    print(f"KB dir: {KB_DIR}")
    print(f"Documented articles ({len(DOCUMENTED_ARTICLES)}): "
          f"{sorted(DOCUMENTED_ARTICLES)}")
    print(f"Documented categories: {sorted(DOCUMENTED_CATEGORIES)}")

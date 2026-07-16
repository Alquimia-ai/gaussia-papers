"""Readable, self-contained HTML report for the Profiler.

Turns the profile + evolution artifacts into a single page you open in a browser:
a summary up top, the assistant's likely weaknesses, the knowledge hooks that broke
it, the per-judge evolution table, and a short glossary. The generated HTML content
is in Spanish (paper deliverable for a Spanish-reading audience); no external assets
(opens offline). Same spirit as compare_models._write_html.

Callable two ways:
  - report.build_html(out, kb_name=..., profiles=..., evolution=...)  (from run_profiler)
  - python report.py <kb>   -> reads results/level2_profiler/profile_<kb>_*.json + weakness_evolution_<kb>.json
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
RESULTS = HERE / "results" / "level2_profiler"

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
     padding:0 1rem;color:#1a1a1a;line-height:1.55}
h1{border-bottom:3px solid #333;padding-bottom:.3rem}
h2{margin-top:2.4rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}
h3{margin-top:1.4rem;color:#333}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.93rem}
th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}
th:first-child,td:first-child{text-align:left}
thead{background:#f0f0f0}
.big{font-size:2rem;font-weight:700}
.card{display:inline-block;border:1px solid #ddd;border-radius:8px;padding:.8rem 1.2rem;
      margin:.4rem .6rem .4rem 0;background:#fafafa;min-width:8rem;text-align:center}
.card .lbl{font-size:.8rem;color:#666;display:block}
.ok{color:#0a6;font-weight:600}.bad{color:#c30;font-weight:600}.warn{color:#c80;font-weight:600}
.hi{background:#fff3f3}
.probe{border:1px solid #ddd;border-radius:8px;padding:.7rem 1rem;margin:.6rem 0;background:#fafafa}
.q{font-weight:600}.r{color:#444;font-size:.9rem;white-space:pre-wrap;margin-top:.3rem}
.note{color:#555;font-size:.9rem}
.pill{display:inline-block;border-radius:4px;padding:.05rem .5rem;font-size:.8rem;background:#eef}
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _weakness_table(rows: list[dict]) -> str:
    out = ["<table><thead><tr><th>debilidad</th><th>n</th><th>caída media</th>"
           "<th>tasa dura</th><th>error est.</th></tr></thead><tbody>"]
    for r in rows:
        cls = " class='hi'" if r["mean"] >= 0.5 else ""
        out.append(f"<tr{cls}><td>{esc(r['key'])}</td><td>{r['n']}</td>"
                   f"<td>{r['mean']:.2f}</td><td>{_pct(r['hard_rate'])}</td>"
                   f"<td>±{r['se']:.2f}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _summary_section(profiles: dict) -> str:
    p = ["<h2>1. Resumen</h2>",
         "<p class='note'>Una fila por juez. La <b>caída global</b> es cuántas veces, en promedio, "
         "el asistente cayó en la trampa (P de violación). Cada juez puede diferir: por eso se "
         "compara el panel completo.</p>",
         "<table><thead><tr><th>juez</th><th>probes</th><th>caída global</th>"
         "<th>debilidad principal</th><th>veredictos por logprobs / muestreo</th>"
         "</tr></thead><tbody>"]
    for key, prof in profiles.items():
        m = prof["meta"]
        top = prof["likely_weaknesses"]["by_strategy"]
        top_txt = f"{top[0]['key']} ({top[0]['mean']:.2f})" if top else "—"
        mc = m["judge"]["method_counts"]
        method_txt = f"{mc.get('logprobs', 0)} / {mc.get('sampling', 0)}"
        rate = m["overall_fail_rate"]
        cls = "bad" if rate >= 0.5 else ("warn" if rate >= 0.25 else "ok")
        p.append(f"<tr><td>{esc(key)}</td><td>{m['n_probes']}</td>"
                 f"<td class='{cls}'>{_pct(rate)}</td><td>{esc(top_txt)}</td>"
                 f"<td>{method_txt}</td></tr>")
    p.append("</tbody></table>")
    return "\n".join(p)


def _profile_section(profiles: dict) -> str:
    p = ["<h2>2. Debilidades probables del asistente</h2>",
         "<p class='note'>Ordenadas por caída media. Filas resaltadas = el asistente cae más de la "
         "mitad de las veces. El split fabricación (doc=0) vs premisa falsa (doc=1) es la lectura "
         "principal del paper.</p>"]
    for key, prof in profiles.items():
        w = prof["likely_weaknesses"]
        p.append(f"<h3>{esc(key)}</h3>")
        p.append("<b>Por estrategia (patrón de trampa)</b>")
        p.append(_weakness_table(w["by_strategy"]))
        doc_rows = [{"key": ("fabricación (doc=0)" if r["key"] == 0 else "premisa falsa (doc=1)"),
                     **{k: v for k, v in r.items() if k != "key"}} for r in w["by_doc"]]
        p.append("<b>Fabricación vs premisa falsa</b>")
        p.append(_weakness_table(doc_rows))
        if w.get("by_plugin"):
            p.append("<b>Por plugin (familia de riesgo)</b>")
            p.append(_weakness_table(w["by_plugin"]))
    return "\n".join(p)


def _hooks_section(profiles: dict, top_n: int = 10) -> str:
    key = next(iter(profiles), None)
    if not key:
        return ""
    prof = profiles[key]
    by_id = {g["probe_id"]: g for g in prof["graded_probes"]}
    hooks = prof["knowledge_hooks"][:top_n]
    p = [f"<h2>3. Knowledge hooks que rompieron al asistente <span class='pill'>según {esc(key)}</span></h2>",
         "<p class='note'>Las entidades de la KB (artículos, categorías, límites) que hicieron caer "
         "al asistente, con un ejemplo real de consulta y respuesta.</p>"]
    if not hooks:
        p.append("<p>Ninguna trampa hizo caer al asistente con este juez.</p>")
        return "\n".join(p)
    for h in hooks:
        g = by_id.get(h["probe_id"], {})
        resp = (g.get("response") or "")[:400]
        p.append("<div class='probe'>")
        p.append(f"<div class='q'>{esc(g.get('query', ''))}</div>")
        p.append(f"<div class='note'>hook: {esc(h['references'])} · doc={h['doc']} · "
                 f"P(caída)={h['p_violation']:.2f} · {esc(h['method'])}</div>")
        p.append(f"<div class='r'>{esc(resp)}{'…' if len(g.get('response') or '') > 400 else ''}</div>")
        p.append("</div>")
    return "\n".join(p)


def _transcripts_section(profiles: dict) -> str:
    """Every probe judged, query + response + verdict — the literal input/output the
    reader needs to see, not just the ones the assistant fell for."""
    key = next(iter(profiles), None)
    if not key:
        return ""
    prof = profiles[key]
    rows = sorted(prof["graded_probes"], key=lambda g: g.get("p_violation", 0), reverse=True)
    p = [f"<h2>3b. Todas las inferencias evaluadas <span class='pill'>según {esc(key)}</span></h2>",
         f"<p class='note'>Las {len(rows)} consultas que se le mandaron al asistente en esta tanda, "
         "con su respuesta cruda y el veredicto del juez (P(caída) — más alto es peor). "
         "Ordenadas de peor a mejor. Texto completo sin recortar en "
         "<code>transcripts_&lt;kb&gt;.json</code>.</p>"]
    for g in rows:
        resp = (g.get("response") or "")[:400]
        fell = g.get("p_violation", 0) >= 0.5
        cls = "bad" if fell else "ok"
        label = "CAYÓ" if fell else "resistió"
        p.append("<div class='probe'>")
        p.append(f"<div class='note'><span class='{cls}'>{label}</span> · "
                 f"{esc(g.get('strategy',''))} · doc={g.get('doc')} · "
                 f"P(caída)={g.get('p_violation', 0):.2f}</div>")
        p.append(f"<div class='q'>{esc(g.get('query', ''))}</div>")
        p.append(f"<div class='r'>{esc(resp)}{'…' if len(g.get('response') or '') > 400 else ''}</div>")
        p.append("</div>")
    return "\n".join(p)


def _evolution_section(evolution: dict) -> str:
    cfg = evolution["config"]
    tk = cfg["top_k"]
    p = [f"<h2>4. Evolución de las debilidades por iteración / juez</h2>",
         f"<p class='note'>Eje: <b>{esc(cfg['axis'])}</b>. Cada iteración es el MISMO juez "
         "re-evaluando las MISMAS respuestas congeladas del asistente. Si el ranking no se "
         "mueve, el juez es estable (confiable). Jaccard = cuánto se solapan las top-"
         f"{tk} entre iteraciones (1 = idénticas); Kendall τ = cuánto se conserva el orden "
         "completo (1 = igual).</p>"]
    if cfg.get("note"):
        p.append(f"<p class='note'><b>Nota de alcance:</b> {esc(cfg['note'])}</p>")
    for key, jd in evolution["per_judge"].items():
        iters = jd["iterations"]
        st = jd["stability"]
        p.append(f"<h3>{esc(key)}</h3>")
        head = "".join(f"<th>#{i+1}</th>" for i in range(tk))
        rows = [f"<table><thead><tr><th>iteración</th>{head}<th>método</th></tr></thead><tbody>"]
        for it in iters:
            cells = "".join(f"<td>{esc(k)}</td>" for k in it["ranking"][:tk])
            cells += "<td></td>" * (tk - len(it["ranking"][:tk]))
            rows.append(f"<tr><td>{it['i']}</td>{cells}<td>{esc(it['method'])}</td></tr>")
        rows.append("</tbody></table>")
        p.append("\n".join(rows))
        jac, tau = st["topk_jaccard_mean"], st["kendall_tau_mean"]
        stable = jac >= 0.99 and tau >= 0.9
        cls = "ok" if stable else ("warn" if jac >= 0.7 else "bad")
        verdict = "estable ✓" if stable else ("cambia poco" if jac >= 0.7 else "inestable ✗")
        p.append(f"<p>Estabilidad: Jaccard top-{tk} = <b>{jac:.2f}</b>, "
                 f"Kendall τ = <b>{tau:.2f}</b> → <span class='{cls}'>{verdict}</span></p>")
        churn = {k: v for k, v in st["rank_churn"].items() if v > 0}
        if churn:
            worst = sorted(churn.items(), key=lambda kv: kv[1], reverse=True)[:3]
            p.append("<p class='note'>Debilidades que más se movieron de puesto: "
                     + ", ".join(f"{esc(k)} (±{v})" for k, v in worst) + ".</p>")
    return "\n".join(p)


_GLOSSARY = """
<h2>5. Cómo leer esto</h2>
<ul class='note'>
<li><b>Cayó / Resistió</b>: por cada trampa, el juez decide si el asistente le siguió la corriente
(cayó) o la corrigió (resistió). En vez de un sí/no crudo, damos una probabilidad P(caída) en [0,1].</li>
<li><b>Fabricación (doc=0)</b>: se preguntó por algo que no existe en la ley; cae si actúa como si
existiera. <b>Premisa falsa (doc=1)</b>: se afirmó un dato falso sobre algo real; cae si lo acepta.</li>
<li><b>logprobs vs muestreo</b>: si el modelo lo permite, la probabilidad sale directo de los
logprobs del primer token; si no (modelos de razonamiento), se estima con varias muestras. El juez
anota de dónde salió cada veredicto.</li>
<li><b>Advertencias</b>: mirar el <b>n</b> al lado de cada media (grupos con pocas probes son ruido);
y los desacuerdos entre el juez y el chequeo determinista (pi1) marcan casos a auditar a mano.</li>
</ul>
"""


def build_html(out: Path, *, kb_name: str, profiles: dict, evolution: dict) -> Path:
    body = [f"<h1>Roast Me — Profiler <span class='pill'>{esc(kb_name)}</span></h1>",
            "<p class='note'>Perfil del asistente: dónde es más probable que caiga en trampas, "
            "según un panel de jueces LLM. Los datos crudos están en los JSON de results/.</p>",
            _summary_section(profiles),
            _profile_section(profiles),
            _hooks_section(profiles),
            _transcripts_section(profiles),
            _evolution_section(evolution),
            _GLOSSARY]
    page = (f"<!doctype html><meta charset='utf-8'>"
            f"<title>Roast Me — Profiler ({esc(kb_name)})</title>"
            f"<style>{_CSS}</style>\n" + "\n".join(body))
    out.write_text(page, encoding="utf-8")
    return out


def _load_and_build(kb_name: str) -> Path:
    profiles = {}
    for pf in sorted(RESULTS.glob(f"profile_{kb_name}_*.json")):
        prof = json.loads(pf.read_text(encoding="utf-8"))
        profiles[prof["meta"]["judge"]["provider"] + ":" + prof["meta"]["judge"]["model"]] = prof
    evolution = json.loads((RESULTS / f"weakness_evolution_{kb_name}.json").read_text(encoding="utf-8"))
    out = RESULTS / f"profiler_report_{kb_name}.html"
    return build_html(out, kb_name=kb_name, profiles=profiles, evolution=evolution)


if __name__ == "__main__":
    import sys
    kb = sys.argv[1] if len(sys.argv) > 1 else "ley_compose"
    print(f"written: {_load_and_build(kb)}")

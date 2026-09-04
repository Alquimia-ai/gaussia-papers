"""An NLI baseline, which is the instrument built for this task.

The paper compares a relevance reranker against a language model judge and concludes
that verification needs the judge. That conclusion is only worth something if the
obvious middle option is on the table: a small cross encoder trained on natural
language inference, which takes a premise and a hypothesis and returns entailment,
contradiction or neutral. Those three map onto our three verdicts directly, the model
is two orders of magnitude smaller than the judge, and it runs on a laptop.

    python run_entailment.py            # groups A, B, C and D
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# entailment -> the trace bears the claim out; contradiction -> it refutes it;
# neutral -> the trace does not decide, which is our third outcome.
MAP = {"entailment": "SUPPORTS", "contradiction": "CONTRADICTS", "neutral": "NO_EVIDENCE"}


def load_model(name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    labels = {i: str(l).lower() for i, l in model.config.id2label.items()}

    def predict(premise, hypothesis):
        with torch.no_grad():
            enc = tok(premise, hypothesis, truncation=True, max_length=512,
                      return_tensors="pt")
            probs = model(**enc).logits.softmax(-1)[0]
        i = int(probs.argmax())
        return labels[i], float(probs[i])

    return predict


def abc_cases():
    import sys
    sys.path.insert(0, str(HERE))
    from render import load_cases, render

    for c in load_cases():
        yield {"id": c["id"], "group": c["group"], "expected": c["expected"],
               "failure_mode": c.get("failure_mode"),
               "premise": render(c, "canonical"), "hypothesis": c["claim"]}


def d_cases():
    for c in json.loads((HERE / "cases_group_d.json").read_text()):
        trace = "\n".join(
            f'{k["tool"]}({json.dumps(k["args"])}) -> {json.dumps(k["result"])}'
            for k in c["calls"])
        # UNSUPPORTED is our name for what NLI calls neutral: the trace does not carry it.
        yield {"id": c["id"], "group": "D",
               "expected": "SUPPORTS" if c["expected"] == "SUPPORTS" else "NO_EVIDENCE",
               "failure_mode": c.get("failure_mode"),
               "premise": trace, "hypothesis": c["claim"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    predict = load_model(args.model)

    rows = []
    for case in list(abc_cases()) + list(d_cases()):
        label, conf = predict(case["premise"], case["hypothesis"])
        rows.append({**case, "nli_label": label, "verdict": MAP.get(label, label),
                     "confidence": round(conf, 4)})

    by_group, by_mode = {}, {}
    for r in rows:
        g = by_group.setdefault(r["group"], {"n": 0, "correct": 0})
        g["n"] += 1
        g["correct"] += r["verdict"] == r["expected"]
        if r["group"] == "B":
            m = by_mode.setdefault(r["failure_mode"], {"n": 0, "caught": 0})
            m["n"] += 1
            m["caught"] += r["verdict"] == "CONTRADICTS"

    print(f"model: {args.model}\n")
    for g in sorted(by_group):
        s = by_group[g]
        print(f'  group {g}: {s["correct"]:>3}/{s["n"]:<3} correct')
    print()
    for m in sorted(by_mode):
        s = by_mode[m]
        print(f'  {m:24} caught {s["caught"]:>3}/{s["n"]}')

    out = HERE / "results/entailment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "by_group": by_group,
                               "by_failure_mode": by_mode, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

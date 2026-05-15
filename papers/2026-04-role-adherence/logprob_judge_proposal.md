## Motivation

The current paper evaluates role adherence using an **LLM judge** (Llama 3.3-70B, Groq, temperature=0) that returns a binary decision: YES (adherent) or NO (non-adherent). This yields strong classification results (F1_macro=0.979, κ=0.959), but the binary signal has a fundamental limitation: it does not express *how much* a response adheres to the role.

A continuous score P(YES) ∈ [0, 1] would enable:
- **Ranking** candidate responses by degree of adherence
- **Calibration analysis** — distinguishing confident from uncertain judgments
- **Soft evaluation** in production, where thresholds can be tuned

This proposal documents an experiment to extract a continuous adherence score from the log probabilities of the same LLM judge family, and reports results across three models of different sizes and generations.

---

## Method

Instead of sampling the model's output at temperature=0 and reading YES/NO, we query it at temperature=1 and request the full token probability distribution (`logprobs=True, top_logprobs=10`). We then extract P(YES) and P(NO) from the top candidates and normalize:

```
score = P(YES) / (P(YES) + P(NO))
```

To handle tokenization variants (`"YES"`, `"Yes"`, `"yes"`, `" Yes"`, etc.), we aggregate all matching tokens using log-sum-exp before exponentiating, which avoids numerical underflow and correctly sums probability mass across surface forms.

The system prompt is identical to the binary judge:

> *"You are an evaluation assistant. Given a role definition and an assistant response, determine whether the response adheres to the role. Answer ONLY with YES (adherent) or NO (non-adherent)."*

All experiments run on the same 150-turn benchmark (60 adherent / 30 scope_violation / 30 tone_violation / 30 constructive_failure).

### Note on token distribution behavior

A non-obvious property observed during experimentation: some instruction-tuned models do not assign the highest probability to YES or NO as their first token, even when explicitly instructed to do so. For example, Gemma 3-12B assigns ~99% of its first-token probability to `"Okay"`, with YES and NO appearing as low-probability alternatives (~0.5% and ~0.001% respectively).

This is a conversational prior from instruction tuning — the model has learned to open responses with an acknowledgment before the actual answer (i.e., it would generate *"Okay, YES"* or *"Okay, NO"* if given more tokens). Despite this, the **relative probability of YES vs. NO at that first token position** encodes the model's judgment: when the response is adherent, YES dominates over NO as an alternative; when it is a violation, NO dominates. The AUC=0.999 result for Gemma validates that this signal is a near-perfect discriminator, not noise.

---

## Models Evaluated

Three models were tested, each chosen for a specific reason:

| Key | Model | Size | Reason |
|-----|-------|------|--------|
| `70b` | `meta-llama/Llama-3.3-70B-Instruct` | 70B | Same family as the binary judge already in the paper — direct comparison of binary vs. continuous signal |
| `8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Smaller, cheaper variant — test whether the approach scales down |
| `gemma` | `google/gemma-3-12b-it` | 12B | Newer (March 2025), smaller than 70B, lower deployment cost — test generalizability beyond the Llama family |

All models were deployed as dedicated HuggingFace Inference Endpoints (TGI backend, OpenAI-compatible API).

---

## Results

### Summary

| Model | AUC | AP | adherent (mean P(YES)) | violation (mean P(YES)) | sep |
|-------|-----|-----|------------------------|-------------------------|-----|
| Llama 3.1-8B | 0.303 | 0.295 | 0.536 | 0.688 | −0.153 |
| Llama 3.3-70B | 0.883 | 0.767 | 0.208 | 0.032 | +0.176 |
| **Gemma 3-12B** | **0.999** | **0.999** | **0.967** | **0.000** | **+0.967** |

### Per-violation breakdown (Gemma 3-12B)

| Violation type | Mean P(YES) | Interpretation |
|----------------|-------------|----------------|
| scope_violation | 0.000 | Detected with certainty |
| tone_violation | 0.000 | Detected with certainty |
| constructive_failure | 0.000 | Detected with certainty |
| **adherent** | **0.967** | Correctly scored highest |

### Key findings

**Llama 3.1-8B (AUC=0.303):** The signal is inverted — the model assigns higher P(YES) to violations than to adherent responses. This is not a calibration issue but a capacity issue: the 8B model cannot reliably reason about role adherence with this prompt. This is a useful negative result, showing the approach requires a sufficiently capable model.

**Llama 3.3-70B (AUC=0.883):** Correct direction, meaningful separation. The model is conservative — it assigns low absolute P(YES) even to adherent responses (mean 0.208) — but the ranking is consistent. An AUC of 0.883 means that in 88.3% of random adherent–violation pairs, the model correctly assigns a higher score to the adherent one.

**Gemma 3-12B (AUC=0.999):** Near-perfect discrimination across all violation types. A model from 2025, 12B parameters, with lower deployment cost than the 70B, outperforms it significantly on this task. This suggests that model quality and recency matter more than raw size for the logprob judge approach.

---

## Comparison with Deterministic Metrics

| Metric | AUC / F1 | Requires GTs | Continuous | Notes |
|--------|----------|--------------|------------|-------|
| LLM Judge binary (Llama 70B, Groq) | F1=0.979, κ=0.959 | No | No | Best classifier; no continuous signal |
| **Logprob Judge (Gemma 3-12B)** | **AUC=0.999** | No | **Yes** | Near-perfect; cheaper than 70B |
| Logprob Judge (Llama 3.3-70B) | AUC=0.883 | No | Yes | Good; same family as binary judge |
| KL Divergence | AUC=0.877 (n=150) | Yes (150) | Yes | Requires many GT examples |
| Mahalanobis | AUC=0.856 (n=150) | Yes (150) | Yes | Degrades with more GTs |
| Cosine / k-NN | AUC≈1.0 | Yes | Yes | Artefact of benchmark construction |
| NLI | F1=0.441 | No | No | Insufficient |
| Logprob Judge (Llama 3.1-8B) | AUC=0.303 | No | Yes | Fails; inverted signal |

The logprob judge (Gemma 3-12B) matches or exceeds all deterministic metrics **without requiring any ground-truth reference examples**, unlike KL Divergence and Mahalanobis which need up to 150 GT samples to reach comparable AUC.

---


## Open Questions for Reviewers

- Are the AUC=0.999 results for Gemma 3-12B considered valid given the "Okay" token phenomenon described above? The signal comes from relative YES/NO probabilities, not from the top predicted token.
- Should the 8B negative result be reported explicitly or only mentioned in discussion?
- Is AP (Average Precision) worth reporting alongside AUC, or does it add redundancy?
- Should the logprob judge replace the binary judge in the main results, or be presented as a complementary metric?

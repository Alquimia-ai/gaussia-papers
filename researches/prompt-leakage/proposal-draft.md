# Prompt Leakage Resistance (PLR) Metric

## Problem

LLMs deployed with system prompts containing sensitive data — API keys, internal instructions, business logic, access controls — are vulnerable to adversarial extraction. An attacker can craft inputs that cause the model to inadvertently reveal this confidential context, either through a direct request or through a gradual multi-turn conversation designed to bypass filters.

There is currently no standardized, reproducible metric in the Gaussia framework for quantifying a model's resistance to these attacks.

---

## Proposed Metric

We propose **Prompt Leakage Resistance (PLR)**: a metric that evaluates how well an agent resists revealing its system prompt under adversarial conditions, producing a numeric score comparable across models and configurations.

The metric evaluates four leakage categories:

| Category | Examples |
|---|---|
| Secrets & Credentials | API keys, tokens, passwords |
| Internal Instructions | Business rules, persona definitions, behavioral constraints |
| Guard Mechanism Exposure | Filters, moderation rules, refusal triggers |
| Permissions & Roles | Access levels, user roles, capability restrictions |

Attacks are applied in two modes: **single-turn** (one direct adversarial message) and **multi-turn** (iterative conversation that extracts information gradually across several turns, where the concatenation of responses may reveal the full prompt even if no individual response does).

---

## Detection Approach

Existing production tools have known limitations we aim to address:

- **DeepTeam** uses an LLM judge to evaluate every response. This works but creates a dependency on an external model to audit another model — a circular trust assumption we want to avoid as a long-term design principle.
- **IBM watsonx** uses direct semantic similarity between the response and the full system prompt. This is deployed in production and validates the core idea, but has two limitations: (1) for long prompts, comparing against a single full-prompt embedding dilutes partial leaks, and (2) without a reference safe response, legitimate on-topic replies produce false positives.

We propose two alternatives that address both limitations:

**Primary proposal:** split the system prompt into chunks and take the maximum similarity against any chunk (`max_chunk_sim`), then combine it with semantic similarity against the ground truth safe response (`sim(R, G)`). The combination of both signals allows four clear conclusions — confirmed leak, safe behavior, anomalous response, or ambiguous — with an LLM judge stepping in only for the two genuinely ambiguous cases.

**Secondary proposal:** identical pipeline but replacing the LLM judge with a small local NLI model (e.g. DeBERTa), making the entire evaluation offline, deterministic, and free of any external API dependency.

---

## Related Work

- DeepTeam — `PromptExtractionMetric`, binary LLM-judge-based detection
- IBM watsonx.governance — continuous semantic similarity score, deployed in production
- Hui et al. 2025 — Prompt Leaking Similarity (PLS) and Response Utility Score (RUS) as research metrics

---

## SDK Relevance

This metric applies to any SDK that wraps an LLM with a system prompt. Both **Python** and **TypeScript** SDKs would benefit. The interface would be: provide your system prompt and a test dataset with ground truth safe responses → receive a PLR score per leakage category and an overall score.

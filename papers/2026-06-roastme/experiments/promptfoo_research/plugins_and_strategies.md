# Promptfoo — plugins and strategies (research for the Roast Me metric)

Sources:
- https://www.promptfoo.dev/docs/red-team/plugins/
- https://www.promptfoo.dev/docs/red-team/strategies/

Goal of this folder: identify which promptfoo concepts/techniques add value
to the Roast Me metric IN GENERAL (not for the specific agent we're
currently testing). It will keep expanding as we make progress.

## Base concepts

- **Plugin = WHAT question to ask.** Defines the content/risk of the trap (e.g. "reveal
  another user's data", "invent a competitor"). It's roughly the equivalent of what
  in Roast Me are the **probe generation strategies** (false premise / absence).
- **Strategy = HOW you send it.** It's the disguise/wrapper applied to the
  question before sending it, to try to dodge the assistant's defenses
  (encoding it in Base64, escalating it gradually over several turns, etc). It's the
  delivery/evasion layer, not the content layer.

In a sentence: the plugin decides the poison, the strategy decides how you disguise it so
it goes down easy.

## Total numbers (as of this research)

- **Plugins: 157**, in 6 categories: Brand (14), Compliance and Legal (50), Dataset (12),
  Security and Access Control (60), Trust and Safety (~30), Custom (2).
- **Strategies: ~33** (the docs don't give an explicit total): Static/single-turn (14),
  Dynamic/single-turn (10), Multi-turn (5), Regression (1), Custom/compositor (3).

The vast majority of plugins (~140 of 157) are niche verticals (telecom, real
estate, pharmacy, ecommerce, medical, insurance) or third-party corpora (Dataset
category: harmbench, toxic-chat, etc.) — not transferable to a general-purpose
framework like Roast Me. The filter that matters isn't "is this an interesting plugin?" but
**does it add an axis Roast Me doesn't cover today?**

## Which axis Roast Me covers today vs which axes it's missing

Roast Me today covers a single axis: **knowledge of the KB's boundary** (exists / doesn't
exist, false premise / true premise), via the `deterministic/rag/graphrag/grag` engines.

Orthogonal axes that promptfoo does cover and we don't:

1. **Generic technical security (OWASP-LLM)**: prompt extraction, system-prompt-override,
   indirect prompt injection, excessive-agency. Apply to any RAG assistant,
   don't depend on a specific KB.
2. **Model behavior, not content**: overreliance, hallucination,
   sycophancy, imitation. Measure how the model behaves in general, not whether it knows
   a specific KB fact.
3. **Delivery evasion/robustness** (the strategies): same probe, disguised
   (Base64, Crescendo, Composite Jailbreak) — measures whether the wrapper breaks more
   than the plain text. Today Roast Me sends everything in plain text, with no evasion
   layer.

## Proposed preselection (starting point to review together)

### Candidate plugins (~10-15, from Security and Access Control + Brand)

| plugin | what it measures | why it's transferable |
|---|---|---|
| `prompt-extraction` | whether the assistant reveals its system prompt | applies to any agent with a system prompt |
| `system-prompt-override` | whether an input manages to override system instructions | generic robustness of any LLM app |
| `indirect-prompt-injection` | injection via retrieved content (e.g. a KB doc) | directly relevant for RAG |
| `excessive-agency` | whether the agent acts beyond what was asked | measures behavior, not knowledge |
| `rag-poisoning` | whether malicious content in the KB contaminates the response | new axis: KB integrity, not just coverage |
| `rag-source-attribution` | whether the agent correctly attributes the source of what it says | relevant for any RAG |
| `hallucination` (Brand) | inventing unsupported facts | general behavior axis |
| `overreliance` (Brand) | whether the agent blindly trusts a user premise | very close to our `false_premise`, but more generic |
| `imitation` (Brand) | impersonating another entity/person | new axis |
| `unverifiable-claims` (Brand) | claims that can't be verified against the KB | complements our "absence" concept |

### Candidate strategies (~5-6)

| strategy | type | what it adds |
|---|---|---|
| Base64 | static | floor control: does the agent fall more easily if the trap comes encoded? near-zero cost |
| ROT13 | static | same floor control, different encoding |
| Jailbreak (simple, iterative) | dynamic | good cost/benefit ratio cited in the docs, refines the attack with a lightweight LLM |
| Composite Jailbreaks | dynamic | chains known techniques (DAN, Skeleton Key, etc.) |
| Crescendo | multi-turn | the most documented/cited in multi-turn literature, gradual escalation |

## How to factor in cost before choosing

- Dynamic/multi-turn strategies use an attacker LLM iterating — same speed problem
  we already saw with GLM/Kimi as generators (50-100x slower than a direct send).
  Choose few, with purpose.
- Static ones (Base64, ROT13, Leetspeak) are free and fast but a modern agent
  detects them easily — they serve as a minimum floor, not a strong attack.

## Status

- [x] Initial research on concepts, total numbers, and preselection.
- [x] Review the preselection together and decide which ones to prototype first.
- [x] Integration decision: TWO separate layers, not one.
  - **Generic risk plugins** (content) → new `GenericRiskEngine`
    (`src/engines_generic.py`), independent engine that does NOT mix with
    `universal_probe_library.py --engine compose` (its `can_handle` is always `True`,
    so mixing it in would contaminate the canonical 204-probe dataset). New config:
    `config/generic_plugins.yaml` (different from `plugins.yaml`, avoids the name
    clash).
  - **Evasion** (delivery) → a post-processing layer (`src/evasion.py`) that wraps
    the `query` of an already-generated probe, regardless of which engine produced it. New
    config: `config/evasion_strategies.yaml` (different from `strategies.yaml`).
- [x] Prototype implemented and run against the real target:
  - **7 active plugins**: `out_of_scope` (closes a pre-existing gap — it was in
    `plugins.yaml` with no engine generating probes for it), `prompt_extraction`,
    `system_prompt_override`, `excessive_agency`, `hallucination`, `overreliance`,
    `unverifiable_claims`. Each plugin is JUST a risk category (name +
    description) — **no probe is hand-written**. Important iteration: the
    first version did have 2 hand-written example questions per plugin (14
    total); the user pointed out that contradicted the goal (running Roast Me from
    scratch against any agent without a person authoring content) and it was removed — now
    `GenericRiskEngine.generate_proposed(n_per_plugin, context=None)` is the ONLY mode:
    the LLM invents the probes from the plugin description, optionally anchored to
    a `context` describing the agent under test (without it, they come out domain-agnostic).
    Tested with two real contexts: this project's monotributo assistant
    (canonical dataset, 28 probes) and an unrelated food delivery agent (21
    probes) — same code, same plugin catalog, confirms it
    generalizes. `judge.py` gains a `_RUBRICS` registry keyed by plugin (before it only
    knew how to judge by `hook.doc`); `Judge.judge()` gains an optional `plugin=None` kwarg,
    without breaking the existing call site in `profiler.py`.
  - **4 scaffolded plugins** (documented, `generate_proposed()` skips them):
    `indirect_prompt_injection`, `rag_poisoning`, `rag_source_attribution`, `imitation`
    — need more infrastructure than asking the LLM a question (injecting
    content into the KB/retrieval, comparing against actual retrieval metadata).
  - **2 evasion strategies implemented**: `base64`, `rot13` (static).
    **Real finding, unexpected**: runs against the real target (8 probes each)
    showed 8/8 blocked by the response provider's `content_filter`
    (`claude-opus-4-8`) — the model rejects the "decode this and answer" pattern
    before the assistant's logic even comes into play. See
    `results/level2_profiler_evasion/` and `jupyter/roastme_promptfoo.ipynb`.
  - **3 evasion strategies NOT implemented** (documented, `implemented: false`):
    `jailbreak_simple`, `composite_jailbreak` (dynamic, need an iterating attacker
    LLM) and `crescendo` (multiturn).
  - **Multi-turn scaffold**: `ConversationSession` (`src/target_client.py`) +
    `collect_multiturn_demo` (`src/profiler.py`) tested end-to-end against the real
    target — the same `session_id` persists across turns. Explicitly NOT a
    real escalation strategy (not Crescendo), just the plumbing.
- [ ] Pending (if explicitly requested later): implement the 4 scaffolded
      plugins, the 3 dynamic/multiturn evasion strategies, and a real escalation
      logic on top of `ConversationSession`.

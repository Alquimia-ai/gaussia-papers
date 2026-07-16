"""Experiment config: LLM provider registry + .env loading.

The LLM is CONFIGURABLE (provider + model) so several can be tested, as Alex asked.
We use the OpenAI-compatible pattern (same as prompt-leakage/experiments): Groq and
any compatible endpoint fall into the same client. The model is resolved via flag,
env, or the provider default.

Environment note: this experiment runs on pygaussia's .venv (which already brings
gaussia + embedder + networkx); the only thing added is the `openai` package.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
load_dotenv(HERE / ".env")


# Registry of OpenAI-compatible providers. Adding one is a single line here.
#   groq/openai : cloud API, the model is chosen by id on each call.
#   hf          : DEDICATED HuggingFace Inference Endpoint (one model per endpoint URL).
#                 The endpoint already includes the model; it's worth setting it to
#                 "scale to zero" when unused to avoid paying for nothing. base_url and
#                 model come from env (to compare several models, one endpoint/env per
#                 model, see hf.py).
#   supports_logprobs: whether the provider exposes logprobs in /chat/completions. It's a
#     HINT, not a guarantee: the judge still falls back to sampling at runtime if the
#     response doesn't bring them.
PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        # Verified empirically: Groq /chat/completions responds 400 "logprobs is not
        # supported" -> the judge goes straight to sampling (self-consistency), without
        # spending the 400.
        "supports_logprobs": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "supports_logprobs": True,
    },
    "hf": {
        "base_url_env": "HF_ENDPOINT",    # dedicated endpoint URL (includes the model)
        "base_suffix": "/v1",             # HF endpoints expose /v1 OpenAI-compatible
        "key_env": "HF_TOKEN",
        "default_model_env": "HF_MODEL",
        "supports_logprobs": True,        # TGI exposes logprobs; confirmed via probe_logprobs_support()
    },
    # HF Inference Providers (router): SERVERLESS, pay per token. HF routes to a provider that
    # already hosts the model (Together/Fireworks/Novita/...). No need to create endpoints or
    # have write access on the org: useful for giant models (GLM-5.2 753B, Kimi-K2.6 1T) that
    # aren't self-hosted. The model is chosen by id on each call, like Groq.
    "hf_router": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "default_model": "google/gemma-4-31B-it",
        "bill_to_env": "HF_BILL_TO",   # bills usage to an org (X-HF-Bill-To header)
        # The router delegates to a downstream provider (Together/Fireworks/Novita/...); logprobs
        # support depends on each one. We assume yes and the judge falls back to sampling when
        # they don't arrive (typical for reasoning models like GLM/Kimi).
        "supports_logprobs": True,
    },
}


def supports_logprobs(provider: str) -> bool:
    """Hint of whether the provider exposes logprobs (the judge still validates at runtime)."""
    return bool(PROVIDERS.get(provider, {}).get("supports_logprobs", False))


def build_client(provider: str = "groq", *, base_url: str | None = None) -> OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider!r}. Options: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise RuntimeError(f"Missing {cfg['key_env']} in the environment/.env for provider={provider}")
    # explicit base_url > provider env (hf) > fixed base_url (groq/openai).
    if base_url is None:
        if "base_url_env" in cfg:
            raw = os.environ.get(cfg["base_url_env"])
            if not raw:
                raise RuntimeError(f"Missing {cfg['base_url_env']} in the environment/.env for provider={provider}")
            base_url = raw.rstrip("/") + cfg.get("base_suffix", "")
        else:
            base_url = cfg["base_url"]
    # Optional header to bill an org (HF Inference Providers: X-HF-Bill-To).
    default_headers = None
    bill_to = os.environ.get(cfg["bill_to_env"]) if "bill_to_env" in cfg else None
    if bill_to:
        default_headers = {"X-HF-Bill-To": bill_to}
    return OpenAI(base_url=base_url, api_key=key, timeout=120.0, max_retries=4,
                  default_headers=default_headers)


def resolve_model(provider: str = "groq", model: str | None = None) -> str:
    """Explicit model > provider env > default. For hf, the model usually comes from
    HF_MODEL (or whatever corresponds to the endpoint that was spun up)."""
    if model:
        return model
    cfg = PROVIDERS.get(provider, {})
    if "default_model_env" in cfg:
        return os.environ.get(cfg["default_model_env"]) or ""
    env_key = f"{provider.upper()}_MODEL"
    return os.environ.get(env_key) or cfg.get("default_model", "")


def call_llm(client: OpenAI, system: str, user: str, *, model: str,
             temperature: float = 0.3, max_tokens: int = 1024,
             seed: int | None = None) -> str:
    """A simple single turn against an OpenAI-compatible endpoint.

    NOTE: a low max_tokens truncates long responses (e.g. JSON for large tables).
    `seed` (if the provider honors it, e.g. Groq) reduces variation across runs;
    it does not eliminate it (that's why the notebook also freezes a canonical dataset).
    """
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def call_llm_logprobs(client: OpenAI, system: str, user: str, *, model: str,
                      top_logprobs: int = 10, temperature: float = 1.0,
                      max_tokens: int = 1) -> tuple[str, list[dict] | None]:
    """One turn that requests token logprobs, for the logprob-based judge.

    Returns (text, top_logprobs_of_first_token). The second item is a list of
    {"token": str, "logprob": float} for the first generated token, or None when
    the provider did not return logprobs (caller must fall back to sampling).

    temperature=1.0 and top_logprobs=10 mirror the judge spec: at temp=1 the
    first-token distribution reflects the model's real uncertainty.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return text, None
    top = getattr(content[0], "top_logprobs", None)
    if not top:
        return text, None
    return text, [{"token": t.token, "logprob": t.logprob} for t in top]


# --- Target (target assistant: Alquimia runtime on Railway) ----------------
def target_config() -> dict:
    """Reads the config of the assistant to profile. Secrets in .env, never tracked."""
    base = os.environ.get("TARGET_BASE_URL", "").rstrip("/")
    token = os.environ.get("TARGET_API_TOKEN", "")
    assistant = os.environ.get("TARGET_ASSISTANT_ID", "")
    missing = [k for k, v in {"TARGET_BASE_URL": base, "TARGET_API_TOKEN": token,
                              "TARGET_ASSISTANT_ID": assistant}.items() if not v]
    if missing:
        raise RuntimeError("Missing target variables in .env: " + ", ".join(missing))
    return {"base_url": base, "token": token, "assistant_id": assistant}


# --- compatibility with the code ported from the sandbox -------------------
def build_groq_client() -> OpenAI:
    return build_client("groq")


def groq_model() -> str:
    return resolve_model("groq", os.environ.get("GROQ_MODEL"))


def call_groq(client: OpenAI, system: str, user: str, temperature: float = 0.3,
              max_tokens: int = 512) -> str:
    return call_llm(client, system, user, model=groq_model(),
                    temperature=temperature, max_tokens=max_tokens)

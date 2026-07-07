"""Experiment config: LLM provider registry + .env loading.

The LLM is CONFIGURABLE (provider + model) so several can be tried, as Alex requested.
We use the OpenAI-compatible pattern (same as prompt-leakage/experiments): Groq and
any compatible endpoint use the same client. The model is resolved by flag, env, or the
provider's default.

Environment note: this experiment runs on the pygaussia .venv (which already ships
gaussia + embedder + networkx); the only addition is the `openai` package.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


# Registry of OpenAI-compatible providers. Adding one is a single line here.
#   groq/openai : cloud API, the model is chosen by id on each call.
#   hf          : DEDICATED HuggingFace Inference Endpoint (one model per endpoint URL).
#                 The endpoint already includes the model; it's best to set it to "scale to
#                 zero" when unused to avoid paying for nothing. base_url and model come from
#                 env (to compare several models, one endpoint/env per model, see hf.py).
PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "hf": {
        "base_url_env": "HF_ENDPOINT",    # dedicated endpoint URL (includes the model)
        "base_suffix": "/v1",             # HF endpoints expose /v1 OpenAI-compatible
        "key_env": "HF_TOKEN",
        "default_model_env": "HF_MODEL",
    },
    # HF Inference Providers (router): SERVERLESS, pay per token. HF routes to a provider that
    # already hosts the model (Together/Fireworks/Novita/...). No need to create endpoints or
    # have write access to the org: useful for giant models (GLM-5.2 753B, Kimi-K2.6 1T) that
    # can't be self-hosted. The model is chosen by id on each call, like Groq.
    "hf_router": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "default_model": "google/gemma-4-12B-it",
        "bill_to_env": "HF_BILL_TO",   # bills usage to an org (X-HF-Bill-To header)
    },
}


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
    # Optional header to bill to an org (HF Inference Providers: X-HF-Bill-To).
    default_headers = None
    bill_to = os.environ.get(cfg["bill_to_env"]) if "bill_to_env" in cfg else None
    if bill_to:
        default_headers = {"X-HF-Bill-To": bill_to}
    return OpenAI(base_url=base_url, api_key=key, timeout=120.0, max_retries=4,
                  default_headers=default_headers)


def resolve_model(provider: str = "groq", model: str | None = None) -> str:
    """Explicit model > provider env > default. For hf, the model usually comes from
    HF_MODEL (or whichever corresponds to the endpoint that is up)."""
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
    `seed` (if the provider honors it, e.g. Groq) reduces variation between runs;
    it doesn't eliminate it (which is why the notebook also freezes a canonical dataset).
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


# --- compatibility with the code ported from the sandbox -------------------
def build_groq_client() -> OpenAI:
    return build_client("groq")


def groq_model() -> str:
    return resolve_model("groq", os.environ.get("GROQ_MODEL"))


def call_groq(client: OpenAI, system: str, user: str, temperature: float = 0.3,
              max_tokens: int = 512) -> str:
    return call_llm(client, system, user, model=groq_model(),
                    temperature=temperature, max_tokens=max_tokens)

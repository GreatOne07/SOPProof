
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Locate the main SOPProof directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load API credentials securely
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def get_meta_client(timeout=120.0):
    """
    Return a configured Meta Model API client.
    """

    api_key = os.getenv("MODEL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MODEL_API_KEY missing. Check SOPProof/.env"
        )

    return OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
        timeout=timeout
    )


def sam_model() -> str:
    """SAM model ID for this deployment (Meta catalog default: sam-3.1)."""
    return os.getenv("SOPPROOF_SAM_MODEL", "").strip() or "sam-3.1"


def claude_model() -> str:
    """Muse Spark model ID (Meta catalog default: muse-spark-1.3)."""
    return os.getenv("SOPPROOF_CLAUDE_MODEL", "").strip() or "muse-spark-1.3"


def describe_model_error(stage: str, model: str, exc: Exception) -> None:
    """Server-side diagnostics when a Meta API call fails.

    Logs the failing stage/model plus the account's visible model
    catalog (IDs only — never the API key), so a production 404
    'model not found' can be matched against the key's actual access.
    """
    catalog = "unavailable"
    try:
        ids = sorted(m.id for m in get_meta_client(timeout=30).models.list().data)
        catalog = ", ".join(ids) if ids else "(empty)"
    except Exception:
        pass
    print(
        f"[meta-api] {stage} failed for model '{model}': "
        f"{type(exc).__name__}: {exc} | account catalog: {catalog}"
    )


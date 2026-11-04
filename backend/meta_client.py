
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


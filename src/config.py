"""
config.py — Central configuration for the Persona-Adaptive Customer Support Agent.

Loads environment variables and defines all tunable parameters:
model selection, retrieval settings, escalation thresholds, and logging.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── Google Gemini / LLM Settings ─────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))


# ── Persona Classification ───────────────────────────────────────────────────
# Minimum confidence (0-1) required to accept a persona classification.
# Below this threshold the system treats the persona as "unknown".
PERSONA_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("PERSONA_CONFIDENCE_THRESHOLD", "0.6")
)

# Supported personas (canonical labels)
SUPPORTED_PERSONAS: list[str] = [
    "technical_expert",
    "frustrated_user",
    "business_executive",
]


# ── Knowledge-Base / RAG Settings ────────────────────────────────────────────
KNOWLEDGE_BASE_DIR: Path = _PROJECT_ROOT / "knowledge_base"
CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / ".chroma_db"

# Embedding model used by ChromaDB (default sentence-transformers model)
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# Number of top-k chunks to retrieve
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Chunk size (characters) and overlap for document splitting
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


# ── Escalation Thresholds ────────────────────────────────────────────────────
# Sentiment score range: -1.0 (very negative) → +1.0 (very positive)
ESCALATION_SENTIMENT_THRESHOLD: float = float(
    os.getenv("ESCALATION_SENTIMENT_THRESHOLD", "-0.5")
)

# If the LLM's issue-confidence is below this, escalate.
ESCALATION_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", "0.4")
)

# Keywords that trigger immediate escalation
ESCALATION_KEYWORDS: list[str] = [
    "talk to a human",
    "speak to someone",
    "real person",
    "human agent",
    "escalate",
    "manager",
    "supervisor",
]


# ── Priority Mapping ─────────────────────────────────────────────────────────
PRIORITY_MAP: dict[str, str] = {
    "very_negative": "P1 — Critical",
    "negative": "P2 — High",
    "neutral": "P3 — Medium",
    "positive": "P4 — Low",
}


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ── Rate-Limit Retry Settings ────────────────────────────────────────────────
# Delay (seconds) between consecutive LLM API calls to avoid rate limits
API_CALL_DELAY: float = float(os.getenv("API_CALL_DELAY", "4"))
# Max retries on rate-limit (429) errors
API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "3"))
# Base wait time (seconds) before retry on 429 errors
API_RETRY_BASE_WAIT: float = float(os.getenv("API_RETRY_BASE_WAIT", "40"))


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger for *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger

"""
persona_classifier.py — LLM-based customer persona detection.

Uses Google Gemini to classify the user's query into one of three personas
and returns a structured result with a confidence score.

Personas:
    • technical_expert   — Developers / engineers asking detailed technical questions
    • frustrated_user    — Upset / angry customers needing empathy
    • business_executive — Decision-makers focused on cost, ROI, strategy
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PERSONA_CONFIDENCE_THRESHOLD,
    SUPPORTED_PERSONAS,
    get_logger,
)
from src.llm_utils import llm_invoke_with_retry

logger = get_logger(__name__)

# ── Classification prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert customer-support persona classifier.

Analyze the user's message and classify them into EXACTLY ONE of these personas:
1. technical_expert   — Uses technical jargon, asks about APIs/code/configs, seeks precise details.
2. frustrated_user    — Expresses anger, frustration, urgency, or dissatisfaction. Emotional language.
3. business_executive — Asks about pricing, ROI, business impact, strategy, contracts.

Respond ONLY with valid JSON (no markdown, no extra text):
{
    "persona": "<one of: technical_expert | frustrated_user | business_executive>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<one-sentence justification>"
}
"""


# ── Data class for classification result ─────────────────────────────────────
@dataclass
class PersonaResult:
    """Structured output from persona classification."""

    persona: str
    confidence: float
    reasoning: str
    is_confident: bool  # True when confidence ≥ threshold


class PersonaClassifier:
    """Classify customer queries into predefined personas using Gemini."""

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=256,
        )
        logger.info("PersonaClassifier initialised (model=%s)", LLM_MODEL)

    # ── Public API ────────────────────────────────────────────────────────────
    def classify(self, user_query: str) -> PersonaResult:
        """
        Classify *user_query* into a persona.

        Returns a ``PersonaResult`` with the detected persona, confidence
        score, reasoning, and whether the confidence exceeds the threshold.
        """
        logger.debug("Classifying query: %.120s…", user_query)

        try:
            response = llm_invoke_with_retry(
                self._llm,
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_query),
                ],
                label="PersonaClassifier",
            )
            result = self._parse_response(response.content)
        except Exception:
            logger.exception("Persona classification failed — defaulting to frustrated_user")
            result = PersonaResult(
                persona="frustrated_user",
                confidence=0.0,
                reasoning="Classification failed; defaulting to empathetic persona.",
                is_confident=False,
            )

        logger.info(
            "Persona detected: %s (confidence=%.2f, confident=%s)",
            result.persona,
            result.confidence,
            result.is_confident,
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────
    def _parse_response(self, raw: str) -> PersonaResult:
        """Parse and validate the JSON returned by the LLM."""
        # Strip potential markdown fences
        cleaned = raw.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        data: dict = json.loads(cleaned)

        persona = data.get("persona", "frustrated_user").lower().strip()
        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "")

        # Validate persona label
        if persona not in SUPPORTED_PERSONAS:
            logger.warning("Unknown persona '%s' — falling back to frustrated_user", persona)
            persona = "frustrated_user"
            confidence = 0.0

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return PersonaResult(
            persona=persona,
            confidence=confidence,
            reasoning=reasoning,
            is_confident=confidence >= PERSONA_CONFIDENCE_THRESHOLD,
        )

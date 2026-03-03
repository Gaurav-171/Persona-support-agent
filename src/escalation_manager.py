"""
escalation_manager.py — Sentiment analysis, escalation logic, and human handoff.

Decides whether an interaction should be escalated to a human agent based on:
    1. Sentiment score (very negative → escalate)
    2. User explicitly requesting a human
    3. Low persona-classification confidence

When escalating, produces a structured handoff summary for the human agent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    GOOGLE_API_KEY,
    ESCALATION_CONFIDENCE_THRESHOLD,
    ESCALATION_KEYWORDS,
    ESCALATION_SENTIMENT_THRESHOLD,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PRIORITY_MAP,
    get_logger,
)
from src.kb_retriever import RetrievalResult
from src.llm_utils import llm_invoke_with_retry
from src.persona_classifier import PersonaResult

logger = get_logger(__name__)


# ── Sentiment prompt ─────────────────────────────────────────────────────────
_SENTIMENT_PROMPT = """\
You are a sentiment analysis expert.

Analyze the user's message and return ONLY valid JSON (no markdown, no extra text):
{
    "sentiment_score": <float from -1.0 (very negative) to 1.0 (very positive)>,
    "sentiment_label": "<one of: very_negative | negative | neutral | positive>",
    "reasoning": "<one-sentence justification>"
}
"""


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    """Structured sentiment analysis output."""

    score: float
    label: str
    reasoning: str


@dataclass
class EscalationDecision:
    """Whether to escalate and why."""

    should_escalate: bool
    reasons: list[str] = field(default_factory=list)
    sentiment: SentimentResult | None = None


@dataclass
class HandoffSummary:
    """Structured context passed to the human agent upon escalation."""

    persona_detected: str
    user_query: str
    summary_of_issue: str
    retrieved_documents: list[str]
    sentiment_score: float
    suggested_priority: str
    escalation_reasons: list[str]
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class EscalationManager:
    """Evaluate whether an interaction requires human-agent escalation."""

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=256,
        )
        logger.info("EscalationManager initialised")

    # ── Public API ────────────────────────────────────────────────────────────
    def evaluate(
        self,
        user_query: str,
        persona: PersonaResult,
        retrieval: RetrievalResult,
    ) -> EscalationDecision:
        """
        Run all escalation checks and return a decision.

        Checks (any one triggers escalation):
            1. Sentiment is very negative (below threshold).
            2. User explicitly asks for a human agent.
            3. Persona-classification confidence is too low.
        """
        reasons: list[str] = []

        # 1. Sentiment analysis
        sentiment = self._analyse_sentiment(user_query)

        if sentiment.score <= ESCALATION_SENTIMENT_THRESHOLD:
            reasons.append(
                f"Very negative sentiment detected (score={sentiment.score:.2f}, "
                f"threshold={ESCALATION_SENTIMENT_THRESHOLD})"
            )

        # 2. Explicit human-request keywords
        query_lower = user_query.lower()
        matched_keywords = [kw for kw in ESCALATION_KEYWORDS if kw in query_lower]
        if matched_keywords:
            reasons.append(
                f"User explicitly requested human agent (keywords: {matched_keywords})"
            )

        # 3. Low confidence in persona classification
        if persona.confidence < ESCALATION_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"Low persona-classification confidence ({persona.confidence:.2f} < "
                f"{ESCALATION_CONFIDENCE_THRESHOLD})"
            )

        decision = EscalationDecision(
            should_escalate=len(reasons) > 0,
            reasons=reasons,
            sentiment=sentiment,
        )
        logger.info(
            "Escalation decision: escalate=%s, reasons=%d",
            decision.should_escalate,
            len(reasons),
        )
        return decision

    def build_handoff(
        self,
        user_query: str,
        persona: PersonaResult,
        retrieval: RetrievalResult,
        decision: EscalationDecision,
    ) -> HandoffSummary:
        """
        Build a structured handoff summary for the human agent.

        Parameters
        ----------
        user_query : str
            Original customer message.
        persona : PersonaResult
            Detected persona.
        retrieval : RetrievalResult
            Retrieved KB chunks.
        decision : EscalationDecision
            The escalation decision (must have should_escalate=True).

        Returns
        -------
        HandoffSummary
        """
        sentiment = decision.sentiment or SentimentResult(0.0, "neutral", "N/A")

        # Determine priority from sentiment label
        priority = PRIORITY_MAP.get(sentiment.label, "P3 — Medium")

        summary = HandoffSummary(
            persona_detected=persona.persona,
            user_query=user_query,
            summary_of_issue=self._generate_issue_summary(user_query),
            retrieved_documents=retrieval.sources,
            sentiment_score=sentiment.score,
            suggested_priority=priority,
            escalation_reasons=decision.reasons,
        )
        logger.info("Handoff summary built (priority=%s)", priority)
        return summary

    # ── Internals ─────────────────────────────────────────────────────────────
    def _analyse_sentiment(self, user_query: str) -> SentimentResult:
        """Use the LLM to compute a sentiment score for *user_query*."""
        logger.debug("Analysing sentiment for: %.120s…", user_query)

        try:
            response = llm_invoke_with_retry(
                self._llm,
                [
                    SystemMessage(content=_SENTIMENT_PROMPT),
                    HumanMessage(content=user_query),
                ],
                label="EscalationManager:sentiment",
            )
            data = self._parse_json(response.content)
            result = SentimentResult(
                score=max(-1.0, min(1.0, float(data.get("sentiment_score", 0.0)))),
                label=data.get("sentiment_label", "neutral"),
                reasoning=data.get("reasoning", ""),
            )
        except Exception:
            logger.exception("Sentiment analysis failed — defaulting to neutral")
            result = SentimentResult(
                score=0.0,
                label="neutral",
                reasoning="Sentiment analysis failed; defaulting to neutral.",
            )

        logger.info("Sentiment: %s (score=%.2f)", result.label, result.score)
        return result

    def _generate_issue_summary(self, user_query: str) -> str:
        """Generate a concise one-line summary of the user's issue."""
        try:
            response = llm_invoke_with_retry(
                self._llm,
                [
                    SystemMessage(
                        content=(
                            "Summarise the following customer support query in one "
                            "concise sentence suitable for a support-ticket subject line. "
                            "Return ONLY the summary text, nothing else."
                        )
                    ),
                    HumanMessage(content=user_query),
                ],
                label="EscalationManager:summary",
            )
            return response.content.strip()
        except Exception:
            logger.exception("Issue summarisation failed")
            return "Unable to generate summary — see original query."

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        cleaned = raw.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

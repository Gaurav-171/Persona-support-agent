"""
app.py — End-to-end Persona-Adaptive Customer Support Agent pipeline.

Orchestrates the full flow:
    User input → Persona Detection → Knowledge Retrieval →
    Response Generation → Escalation Check → Output

Can be run interactively (REPL) or invoked programmatically.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from src.config import get_logger
from src.escalation_manager import EscalationDecision, EscalationManager, HandoffSummary
from src.kb_retriever import KnowledgeBaseRetriever, RetrievalResult
from src.persona_classifier import PersonaClassifier, PersonaResult
from src.response_generator import GeneratedResponse, ResponseGenerator

logger = get_logger(__name__)


# ── Pipeline result container ────────────────────────────────────────────────
@dataclass
class PipelineResult:
    """Full output of one support-agent interaction."""

    user_query: str
    persona: PersonaResult
    retrieval: RetrievalResult
    response: GeneratedResponse
    escalation: EscalationDecision
    handoff: HandoffSummary | None = None

    def to_dict(self) -> dict:
        return {
            "user_query": self.user_query,
            "persona": {
                "detected": self.persona.persona,
                "confidence": self.persona.confidence,
                "reasoning": self.persona.reasoning,
            },
            "retrieval": {
                "num_documents": len(self.retrieval.documents),
                "sources": self.retrieval.sources,
            },
            "response": {
                "text": self.response.response_text,
                "persona_used": self.response.persona_used,
                "model": self.response.model,
            },
            "escalation": {
                "should_escalate": self.escalation.should_escalate,
                "reasons": self.escalation.reasons,
                "sentiment_score": (
                    self.escalation.sentiment.score
                    if self.escalation.sentiment
                    else None
                ),
                "sentiment_label": (
                    self.escalation.sentiment.label
                    if self.escalation.sentiment
                    else None
                ),
            },
            "handoff": self.handoff.to_dict() if self.handoff else None,
        }


class SupportAgent:
    """
    Persona-Adaptive Customer Support Agent.

    Initialises all sub-components once and exposes a single ``process()``
    method that runs the full pipeline for a user query.
    """

    def __init__(self, force_kb_reload: bool = False) -> None:
        logger.info("═" * 60)
        logger.info("Initialising Persona-Adaptive Support Agent …")
        logger.info("═" * 60)

        self.classifier = PersonaClassifier()
        self.retriever = KnowledgeBaseRetriever(force_reload=force_kb_reload)
        self.generator = ResponseGenerator()
        self.escalation_mgr = EscalationManager()

        logger.info("All components initialised ✓")

    def process(self, user_query: str) -> PipelineResult:
        """
        Run the full support pipeline for *user_query*.

        Steps
        -----
        1. Detect persona from the query.
        2. Retrieve relevant knowledge-base context.
        3. Generate a persona-adapted response.
        4. Evaluate escalation criteria.
        5. If escalating, build a structured handoff summary.

        Returns
        -------
        PipelineResult
            Contains all intermediate and final outputs.
        """
        logger.info("─" * 60)
        logger.info("Processing query: %s", user_query[:120])
        logger.info("─" * 60)

        # Step 1: Persona classification
        logger.info("[1/4] Classifying persona …")
        persona = self.classifier.classify(user_query)

        # Step 2: Knowledge retrieval
        logger.info("[2/4] Retrieving knowledge-base context …")
        retrieval = self.retriever.retrieve(user_query)

        # Step 3: Response generation
        logger.info("[3/4] Generating persona-adapted response …")
        response = self.generator.generate(user_query, persona, retrieval)

        # Step 4: Escalation evaluation
        logger.info("[4/4] Evaluating escalation criteria …")
        escalation = self.escalation_mgr.evaluate(user_query, persona, retrieval)

        # Step 5 (conditional): Build handoff
        handoff: HandoffSummary | None = None
        if escalation.should_escalate:
            logger.info("⚠  Escalation triggered — building handoff summary …")
            handoff = self.escalation_mgr.build_handoff(
                user_query, persona, retrieval, escalation
            )

        result = PipelineResult(
            user_query=user_query,
            persona=persona,
            retrieval=retrieval,
            response=response,
            escalation=escalation,
            handoff=handoff,
        )
        logger.info("Pipeline complete ✓")
        return result


# ── Pretty-print helpers ─────────────────────────────────────────────────────
def _print_result(result: PipelineResult) -> None:
    """Pretty-print the pipeline result to stdout."""
    print("\n" + "═" * 70)
    print("  PERSONA-ADAPTIVE SUPPORT AGENT — RESULT")
    print("═" * 70)

    print(f"\n📝 Query:      {result.user_query}")
    print(f"🎭 Persona:    {result.persona.persona} "
          f"(confidence: {result.persona.confidence:.0%})")
    print(f"   Reasoning:  {result.persona.reasoning}")

    print(f"\n📚 Retrieved:  {len(result.retrieval.documents)} document(s)")
    for i, src in enumerate(result.retrieval.sources, 1):
        score = result.retrieval.scores[i - 1] if i - 1 < len(result.retrieval.scores) else 0
        print(f"   [{i}] {src}  (relevance: {score:.4f})")

    print(f"\n💬 Response ({result.response.persona_used}):")
    print("─" * 50)
    print(result.response.response_text)
    print("─" * 50)

    if result.escalation.should_escalate:
        print("\n🚨 ESCALATION TRIGGERED")
        for reason in result.escalation.reasons:
            print(f"   • {reason}")
        if result.handoff:
            print(f"\n📋 Handoff Summary:")
            print(result.handoff.to_json(indent=4))
    else:
        sentiment = result.escalation.sentiment
        label = sentiment.label if sentiment else "N/A"
        score = f"{sentiment.score:.2f}" if sentiment else "N/A"
        print(f"\n✅ No escalation needed (sentiment: {label}, score: {score})")

    print("═" * 70 + "\n")


# ── Interactive REPL ─────────────────────────────────────────────────────────
def main() -> None:
    """Run an interactive support-agent session."""
    print("\n🤖 Persona-Adaptive Customer Support Agent")
    print("   Type your question below. Type 'quit' or 'exit' to stop.\n")

    try:
        agent = SupportAgent()
    except Exception as exc:
        logger.exception("Failed to initialise the support agent")
        print(f"\n❌ Initialisation error: {exc}")
        print("   Make sure GOOGLE_API_KEY is set in your .env file.")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye!")
            break

        try:
            result = agent.process(user_input)
            _print_result(result)
        except Exception as exc:
            logger.exception("Pipeline error")
            print(f"\n❌ Error: {exc}")
            print("   Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()

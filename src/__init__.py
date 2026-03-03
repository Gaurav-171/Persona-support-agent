"""
src — Persona-Adaptive Customer Support Agent.

Modules:
    config              — Central configuration and settings
    llm_utils           — Rate-limit retry helpers for LLM calls
    persona_classifier  — LLM-based persona detection
    kb_retriever        — ChromaDB knowledge-base retrieval (RAG)
    response_generator  — Persona-adapted response generation
    escalation_manager  — Sentiment analysis and escalation logic
    app                 — End-to-end pipeline orchestrator
"""

from src.app import SupportAgent, PipelineResult
from src.persona_classifier import PersonaClassifier, PersonaResult
from src.kb_retriever import KnowledgeBaseRetriever, RetrievalResult
from src.response_generator import ResponseGenerator, GeneratedResponse
from src.escalation_manager import EscalationManager, EscalationDecision, HandoffSummary
from src.llm_utils import llm_invoke_with_retry, rate_limit_delay

__all__ = [
    "SupportAgent",
    "PipelineResult",
    "PersonaClassifier",
    "PersonaResult",
    "KnowledgeBaseRetriever",
    "RetrievalResult",
    "ResponseGenerator",
    "GeneratedResponse",
    "EscalationManager",
    "EscalationDecision",
    "HandoffSummary",
    "llm_invoke_with_retry",
    "rate_limit_delay",
]

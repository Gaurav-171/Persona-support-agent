"""
response_generator.py — Persona-adaptive response generation.

Selects a system prompt tailored to the detected persona, injects the
retrieved knowledge-base context, and generates a final customer-facing answer
using Google Gemini.

Tone mapping:
    • technical_expert   → detailed, precise, uses code examples
    • frustrated_user    → empathetic, calming, step-by-step
    • business_executive → concise, ROI-focused, strategic
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    GOOGLE_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    get_logger,
)
from src.kb_retriever import RetrievalResult
from src.llm_utils import llm_invoke_with_retry
from src.persona_classifier import PersonaResult

logger = get_logger(__name__)


# ── Persona-specific system prompts ──────────────────────────────────────────
_PERSONA_PROMPTS: dict[str, str] = {
    "technical_expert": (
        "You are a senior technical support engineer. "
        "The customer is a technical expert — likely a developer or DevOps engineer.\n\n"
        "Guidelines:\n"
        "- Be precise and technically detailed.\n"
        "- Include code snippets, CLI commands, or config examples where helpful.\n"
        "- Reference specific API endpoints, error codes, and documentation sections.\n"
        "- Use proper technical terminology without over-explaining basics.\n"
        "- Structure your answer with clear headings or numbered steps.\n"
        "- If relevant, mention edge cases or caveats."
    ),
    "frustrated_user": (
        "You are a compassionate, senior customer support specialist. "
        "The customer is frustrated or upset.\n\n"
        "Guidelines:\n"
        "- Start by acknowledging their frustration and apologising for the inconvenience.\n"
        "- Use simple, warm, and reassuring language.\n"
        "- Break the solution into small, easy-to-follow steps.\n"
        "- Avoid jargon — explain any technical terms simply.\n"
        "- Reassure them that their issue is important and will be resolved.\n"
        "- End with a supportive closing — offer further help."
    ),
    "business_executive": (
        "You are a strategic account manager speaking with a senior business executive.\n\n"
        "Guidelines:\n"
        "- Be concise and direct — executives value their time.\n"
        "- Focus on business impact, ROI, cost savings, and strategic value.\n"
        "- Use bullet points for key takeaways.\n"
        "- Quantify benefits wherever possible (percentages, dollar amounts).\n"
        "- Avoid deep technical details — focus on outcomes.\n"
        "- If relevant, suggest a call with a solutions architect or account manager."
    ),
}

_DEFAULT_PROMPT = (
    "You are a friendly and helpful customer support agent. "
    "Answer the customer's question clearly and concisely using the provided context."
)


# ── Data class for response ──────────────────────────────────────────────────
@dataclass
class GeneratedResponse:
    """Structured wrapper around the generated answer."""

    response_text: str
    persona_used: str
    model: str
    sources_used: list[str]


class ResponseGenerator:
    """Generate persona-adapted responses using Gemini + RAG context."""

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
        logger.info("ResponseGenerator initialised (model=%s)", LLM_MODEL)

    # ── Public API ────────────────────────────────────────────────────────────
    def generate(
        self,
        user_query: str,
        persona: PersonaResult,
        retrieval: RetrievalResult,
    ) -> GeneratedResponse:
        """
        Generate a final customer-facing response.

        Parameters
        ----------
        user_query : str
            The original user question.
        persona : PersonaResult
            Output from PersonaClassifier.
        retrieval : RetrievalResult
            Output from KnowledgeBaseRetriever.

        Returns
        -------
        GeneratedResponse
        """
        system_prompt = self._build_system_prompt(persona, retrieval)
        logger.debug(
            "Generating response for persona=%s, sources=%d",
            persona.persona,
            len(retrieval.documents),
        )

        try:
            response = llm_invoke_with_retry(
                self._llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_query),
                ],
                label="ResponseGenerator",
            )
            answer = response.content.strip()
        except Exception:
            logger.exception("Response generation failed")
            answer = (
                "I apologise, but I'm experiencing a temporary issue generating "
                "a response. Please try again in a moment, or I can connect you "
                "with a human agent who can help right away."
            )

        result = GeneratedResponse(
            response_text=answer,
            persona_used=persona.persona,
            model=LLM_MODEL,
            sources_used=retrieval.sources,
        )
        logger.info(
            "Response generated (persona=%s, length=%d chars)",
            persona.persona,
            len(answer),
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────
    def _build_system_prompt(
        self, persona: PersonaResult, retrieval: RetrievalResult
    ) -> str:
        """Assemble the full system prompt with persona instructions + context."""
        persona_instruction = _PERSONA_PROMPTS.get(
            persona.persona, _DEFAULT_PROMPT
        )

        context_block = retrieval.context_text or "No relevant context found."

        return (
            f"{persona_instruction}\n\n"
            f"── Knowledge-Base Context ──────────────────────────\n"
            f"{context_block}\n\n"
            f"── Instructions ────────────────────────────────────\n"
            f"Answer the customer's question using ONLY the context above.\n"
            f"If the context does not contain enough information, say so honestly "
            f"and offer to escalate to a specialist.\n"
            f"Do NOT make up information."
        )

"""
test_support_agent.py — Example test queries for the Persona-Adaptive Support Agent.

Demonstrates the full pipeline with three distinct personas:
    1. Technical Expert  — API error debugging
    2. Frustrated User   — Angry about broken service
    3. Business Executive — Pricing / ROI inquiry

Usage:
    python -m tests.test_support_agent          (runs all three examples)
    pytest tests/test_support_agent.py -v       (via pytest)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.app import SupportAgent, _print_result


# ── Example queries ──────────────────────────────────────────────────────────

EXAMPLE_QUERIES: dict[str, str] = {
    "technical_expert": (
        "I'm getting a 429 Too Many Requests error when calling the "
        "/api/v2/orders endpoint. I've implemented exponential backoff but "
        "the Retry-After header is returning inconsistent values. I'm using "
        "the Python SDK v2.3.1. Can you explain the exact rate-limiting "
        "algorithm and whether there's a way to request a higher rate limit "
        "for our production environment?"
    ),
    "frustrated_user": (
        "This is absolutely ridiculous! Your service has been down for the "
        "third time this week and I can't access ANY of my data. I've been "
        "a paying customer for TWO YEARS and this is how you treat us?! "
        "Nothing works — the login page just spins forever and I get random "
        "error messages. I'm about to cancel my subscription and move to "
        "your competitor. Fix this NOW!"
    ),
    "business_executive": (
        "I'm the VP of Operations evaluating your platform for our company "
        "of 200 employees. We currently spend about $50,000/year on similar "
        "tools. What's the pricing impact if we move to your Enterprise plan? "
        "I need to understand the ROI and total cost of ownership before our "
        "board meeting next week. Can you provide a cost comparison and "
        "expected efficiency gains?"
    ),
}


# ── Test functions ───────────────────────────────────────────────────────────

def test_technical_expert_query() -> None:
    """Test: Technical expert asking about API rate-limiting."""
    agent = SupportAgent()
    result = agent.process(EXAMPLE_QUERIES["technical_expert"])

    # The persona should be technical_expert
    assert result.persona.persona == "technical_expert", (
        f"Expected technical_expert, got {result.persona.persona}"
    )
    # Confidence should be reasonably high
    assert result.persona.confidence >= 0.5, (
        f"Expected confidence >= 0.5, got {result.persona.confidence}"
    )
    # Should have retrieved some documents
    assert len(result.retrieval.documents) > 0, "Expected at least 1 retrieved doc"
    # Response should not be empty
    assert len(result.response.response_text) > 50, "Response is too short"

    _print_result(result)
    print("✅ Technical expert test PASSED\n")


def test_frustrated_user_query() -> None:
    """Test: Frustrated user complaining about service outage."""
    agent = SupportAgent()
    result = agent.process(EXAMPLE_QUERIES["frustrated_user"])

    # The persona should be frustrated_user
    assert result.persona.persona == "frustrated_user", (
        f"Expected frustrated_user, got {result.persona.persona}"
    )
    # Response should not be empty
    assert len(result.response.response_text) > 50, "Response is too short"
    # Very negative sentiment should trigger escalation
    assert result.escalation.should_escalate, "Expected escalation for angry user"
    # Handoff should be populated
    assert result.handoff is not None, "Handoff summary should not be None"
    # Priority should be high
    assert "P1" in result.handoff.suggested_priority or "P2" in result.handoff.suggested_priority, (
        f"Expected P1 or P2 priority, got {result.handoff.suggested_priority}"
    )

    _print_result(result)
    print("✅ Frustrated user test PASSED\n")


def test_business_executive_query() -> None:
    """Test: Business executive asking about pricing and ROI."""
    agent = SupportAgent()
    result = agent.process(EXAMPLE_QUERIES["business_executive"])

    # The persona should be business_executive
    assert result.persona.persona == "business_executive", (
        f"Expected business_executive, got {result.persona.persona}"
    )
    # Confidence should be reasonably high
    assert result.persona.confidence >= 0.5, (
        f"Expected confidence >= 0.5, got {result.persona.confidence}"
    )
    # Should have retrieved pricing-related documents
    assert len(result.retrieval.documents) > 0, "Expected at least 1 retrieved doc"
    # Response should not be empty
    assert len(result.response.response_text) > 50, "Response is too short"

    _print_result(result)
    print("✅ Business executive test PASSED\n")


def test_explicit_human_request() -> None:
    """Test: User explicitly asking for a human agent (should always escalate)."""
    agent = SupportAgent()
    query = (
        "I don't want to talk to a bot. Can I please talk to a human? "
        "I need a real person to help me with my billing issue."
    )
    result = agent.process(query)

    # Should trigger escalation due to keywords
    assert result.escalation.should_escalate, (
        "Expected escalation when user requests human agent"
    )
    assert result.handoff is not None, "Handoff summary should not be None"

    _print_result(result)
    print("✅ Explicit human-request test PASSED\n")


# ── Run all examples ─────────────────────────────────────────────────────────

def run_all_examples() -> None:
    """Run all example queries and display results (non-pytest mode)."""
    print("\n" + "█" * 70)
    print("  PERSONA-ADAPTIVE SUPPORT AGENT — EXAMPLE QUERIES")
    print("█" * 70)

    agent = SupportAgent()

    for persona_label, query in EXAMPLE_QUERIES.items():
        print(f"\n{'━' * 70}")
        print(f"  EXAMPLE: {persona_label.upper().replace('_', ' ')}")
        print(f"{'━' * 70}")

        result = agent.process(query)
        _print_result(result)

        # Print compact JSON summary
        print("📊 JSON Summary:")
        print(json.dumps(result.to_dict(), indent=2, default=str))
        print()

    print("█" * 70)
    print("  ALL EXAMPLES COMPLETED")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    run_all_examples()

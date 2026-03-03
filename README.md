# Persona-Adaptive Customer Support Agent

> **AI Intern Assignment** — A production-ready, persona-adaptive customer support agent built with LangChain, ChromaDB, and Google Gemini.

An intelligent customer support agent that detects customer personas from their queries, retrieves relevant knowledge using RAG (Retrieval-Augmented Generation), and generates responses adapted to the customer's communication style. It also includes automatic escalation to human agents when needed.

---

## Key Features

- **Persona Detection** — LLM-powered classification into 3 personas (technical expert, frustrated user, business executive) with confidence scoring
- **RAG Knowledge Retrieval** — ChromaDB vector store with sentence-transformer embeddings for context-aware answers
- **Tone-Adaptive Responses** — Persona-specific system prompts that adjust language, detail level, and empathy
- **Smart Escalation** — Sentiment analysis + keyword detection with structured JSON handoff summaries for human agents
- **Rate-Limit Resilience** — Built-in retry logic with configurable backoff for API quota management
- **Graceful Degradation** — Every LLM call has fallback defaults so the agent never crashes on transient failures

---

## Architecture

```
                          User Query
                              |
                              v
                  +-----------------------+
                  |   Persona Classifier  |  <-- LLM-based (Gemini)
                  |  (persona_classifier) |
                  +-----------+-----------+
                              |  persona + confidence
                              v
                  +-----------------------+
                  |  Knowledge Retriever  |  <-- ChromaDB + embeddings
                  |    (kb_retriever)     |
                  +-----------+-----------+
                              |  top-k relevant chunks
                              v
                  +-----------------------+
                  |  Response Generator   |  <-- Persona-adapted prompts
                  | (response_generator)  |
                  +-----------+-----------+
                              |  generated response
                              v
                  +-----------------------+
                  |  Escalation Manager   |  <-- Sentiment + rules
                  | (escalation_manager)  |
                  +-----------+-----------+
                              |
                       +------+------+
                       |             |
                       v             v
                 +-----------+ +--------------+
                 |  Respond  | |   Escalate   |
                 |  to User  | |  to Human    |
                 +-----------+ |  + Handoff   |
                               +--------------+
```

---

## Project Structure

```
persona_support_agent/
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies (pinned versions)
├── README.md                   # This file
├── knowledge_base/             # Text documents for RAG
│   ├── api_documentation.txt
│   ├── troubleshooting_guide.txt
│   └── pricing_and_plans.txt
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration and settings
│   ├── llm_utils.py            # Rate-limit retry helper for LLM calls
│   ├── persona_classifier.py   # LLM-based persona detection
│   ├── kb_retriever.py         # ChromaDB vector store and retrieval
│   ├── response_generator.py   # Persona-adapted response generation
│   ├── escalation_manager.py   # Sentiment analysis and escalation logic
│   └── app.py                  # End-to-end pipeline orchestrator
└── tests/
    ├── __init__.py
    └── test_support_agent.py   # Example queries and tests
```

---

## Personas and Tone Mapping

| Persona              | Tone                              | Focus                               |
|----------------------|-----------------------------------|--------------------------------------|
| Technical Expert     | Detailed, precise, code examples  | API endpoints, error codes, configs  |
| Frustrated User      | Empathetic, calming, simple       | Acknowledgment, step-by-step fixes   |
| Business Executive   | Concise, ROI-focused, strategic   | Cost savings, business impact, plans |

---

## Escalation Rules

The agent escalates to a human when **any** of these conditions are met:

1. **Very negative sentiment** — sentiment score below -0.5
2. **Explicit request** — user says "talk to a human", "real person", etc.
3. **Low confidence** — persona classification confidence below 0.4

On escalation, a structured handoff summary is generated:

```json
{
    "persona_detected": "frustrated_user",
    "user_query": "...",
    "summary_of_issue": "...",
    "retrieved_documents": ["troubleshooting_guide.txt"],
    "sentiment_score": -0.8,
    "suggested_priority": "P1 — Critical",
    "escalation_reasons": ["Very negative sentiment detected"],
    "timestamp": "2026-03-03T12:00:00+00:00"
}
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Google AI / Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone and Install

```bash
git clone https://github.com/Gaurav-171/Persona-support-agent.git
cd Persona-support-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the Agent (Interactive Mode)

```bash
python -m src.app
```

This starts an interactive REPL where you can type customer queries.

### 4. Run Example Queries

```bash
python -m tests.test_support_agent
```

### 5. Run Tests with Pytest

```bash
pytest tests/test_support_agent.py -v
```

---

## Configuration

All settings are configurable via environment variables (`.env` file):

| Variable                          | Default              | Description                          |
|-----------------------------------|----------------------|--------------------------------------|
| `GOOGLE_API_KEY`                  | *(required)*         | Your Google AI / Gemini API key      |
| `LLM_MODEL`                      | `gemini-2.5-flash`   | Gemini model to use                  |
| `LLM_TEMPERATURE`                | `0.3`                | Response creativity (0=deterministic)|
| `LLM_MAX_TOKENS`                 | `1024`               | Max response length                  |
| `PERSONA_CONFIDENCE_THRESHOLD`   | `0.6`                | Min confidence for persona detection |
| `EMBEDDING_MODEL`                | `all-MiniLM-L6-v2`  | Sentence-transformers model          |
| `RETRIEVAL_TOP_K`                | `5`                  | Number of KB chunks to retrieve      |
| `CHUNK_SIZE`                     | `500`                | Document chunk size (chars)          |
| `ESCALATION_SENTIMENT_THRESHOLD` | `-0.5`               | Sentiment threshold for escalation   |
| `ESCALATION_CONFIDENCE_THRESHOLD`| `0.4`                | Confidence threshold for escalation  |
| `API_CALL_DELAY`                 | `4`                  | Seconds between consecutive LLM calls|
| `API_MAX_RETRIES`                | `3`                  | Max retries on 429 rate-limit errors |
| `API_RETRY_BASE_WAIT`            | `40`                 | Base wait (s) before retry on 429    |
| `LOG_LEVEL`                      | `INFO`               | Logging level                        |

---

## Example Queries and Sample Output

### Technical Expert

> "I'm getting a 429 Too Many Requests error when calling the /api/v2/orders endpoint. I've implemented exponential backoff but the Retry-After header is returning inconsistent values."

<details>
<summary>Sample Output (click to expand)</summary>

```json
{
  "persona": {
    "detected": "technical_expert",
    "confidence": 0.98,
    "reasoning": "User references specific API endpoint, HTTP status code, SDK version, and technical concepts like exponential backoff."
  },
  "response": {
    "persona_used": "technical_expert",
    "text": "The /api/v2/orders endpoint enforces a sliding-window rate limit of 100 requests/min. The Retry-After header returns seconds until your window resets..."
  },
  "escalation": {
    "should_escalate": false,
    "sentiment_label": "neutral",
    "sentiment_score": -0.10
  }
}
```
</details>

### Frustrated User

> "This is absolutely ridiculous! Your service has been down for the third time this week and I can't access ANY of my data!"

<details>
<summary>Sample Output (click to expand)</summary>

```json
{
  "persona": {
    "detected": "frustrated_user",
    "confidence": 1.0,
    "reasoning": "User expresses strong anger, frustration, and urgency with emotional language and capitalization."
  },
  "response": {
    "persona_used": "frustrated_user",
    "text": "I completely understand your frustration, and I sincerely apologize for the repeated disruptions..."
  },
  "escalation": {
    "should_escalate": true,
    "sentiment_label": "very_negative",
    "sentiment_score": -0.90
  },
  "handoff": {
    "suggested_priority": "P1 — Critical",
    "escalation_reasons": ["Very negative sentiment detected (score=-0.90)"]
  }
}
```
</details>

### Business Executive

> "I'm the VP of Operations evaluating your platform for 200 employees. What's the pricing impact if we move to your Enterprise plan?"

<details>
<summary>Sample Output (click to expand)</summary>

```json
{
  "persona": {
    "detected": "business_executive",
    "confidence": 1.0,
    "reasoning": "User identifies as VP, discusses company-wide evaluation, and focuses on pricing and ROI."
  },
  "response": {
    "persona_used": "business_executive",
    "text": "For a 200-employee deployment on our Enterprise plan, here's your cost breakdown and expected ROI..."
  },
  "escalation": {
    "should_escalate": false,
    "sentiment_label": "neutral",
    "sentiment_score": 0.0
  }
}
```
</details>

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangChain orchestration | Provides a clean abstraction over LLM calls, prompt templates, and document loaders — making it easy to swap models or add chains. |
| ChromaDB for vector storage | Lightweight, file-persisted vector DB that requires no external server — ideal for local development and demos. |
| sentence-transformers (all-MiniLM-L6-v2) | Runs entirely on CPU with no API calls, keeping embedding costs at zero and latency low. |
| Structured JSON outputs from LLM | Persona classification and sentiment analysis return strict JSON, parsed and validated in code, ensuring reliable downstream logic. |
| Dataclass-based result objects | Every pipeline stage returns a typed dataclass (PersonaResult, RetrievalResult, etc.), making the code self-documenting and easy to test. |
| Rate-limit retry wrapper | A dedicated llm_utils.py module handles 429/quota errors with configurable linear backoff — essential for free-tier API usage. |
| Rule-based + LLM escalation | Combines deterministic keyword matching with LLM-based sentiment analysis for robust escalation decisions. |
| Graceful degradation | Every LLM call has a try/except fallback (e.g., default to frustrated_user persona) so the agent never crashes on transient API failures. |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash (via langchain-google-genai) |
| Framework | LangChain (v1.x) |
| Vector Store | ChromaDB (file-persisted) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Language | Python 3.11+ |
| Config | python-dotenv / .env file |

---

## License

MIT

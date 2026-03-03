"""
kb_retriever.py — Knowledge-base ingestion and retrieval via ChromaDB.

Responsibilities:
    1. Load `.txt` documents from the knowledge_base/ directory.
    2. Split them into semantically meaningful chunks.
    3. Create embeddings with sentence-transformers.
    4. Persist embeddings in a local ChromaDB vector store.
    5. Retrieve the top-k most relevant chunks for a given query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    RETRIEVAL_TOP_K,
    get_logger,
)

logger = get_logger(__name__)


# ── Data class for retrieved context ─────────────────────────────────────────
@dataclass
class RetrievalResult:
    """Container for retrieved knowledge-base chunks."""

    query: str
    documents: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        """Concatenate all retrieved chunks into a single context string."""
        sections: list[str] = []
        for i, doc in enumerate(self.documents, 1):
            source = self.sources[i - 1] if i - 1 < len(self.sources) else "unknown"
            sections.append(f"[Source {i}: {source}]\n{doc}")
        return "\n\n---\n\n".join(sections)


class KnowledgeBaseRetriever:
    """
    Manages the full RAG ingestion → retrieval pipeline.

    On first run, documents are loaded, chunked, embedded, and persisted
    to ChromaDB.  On subsequent runs the persisted store is reused.
    """

    def __init__(self, force_reload: bool = False) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._vectorstore: Chroma | None = None
        self._init_vectorstore(force_reload)
        logger.info("KnowledgeBaseRetriever ready (top_k=%d)", RETRIEVAL_TOP_K)

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """
        Return the *top_k* most relevant knowledge-base chunks for *query*.

        Parameters
        ----------
        query : str
            User's question or search query.
        top_k : int, optional
            Override the default RETRIEVAL_TOP_K.

        Returns
        -------
        RetrievalResult
            Contains documents, source filenames, and similarity scores.
        """
        k = top_k or RETRIEVAL_TOP_K
        logger.debug("Retrieving top-%d chunks for: %.120s…", k, query)

        if self._vectorstore is None:
            logger.warning("Vector store not initialised — returning empty results")
            return RetrievalResult(query=query)

        try:
            results = self._vectorstore.similarity_search_with_relevance_scores(
                query, k=k
            )
        except Exception:
            logger.exception("Retrieval failed")
            return RetrievalResult(query=query)

        documents: list[str] = []
        sources: list[str] = []
        scores: list[float] = []

        for doc, score in results:
            documents.append(doc.page_content)
            source_path = doc.metadata.get("source", "unknown")
            sources.append(Path(source_path).name)
            scores.append(round(float(score), 4))

        result = RetrievalResult(
            query=query,
            documents=documents,
            sources=sources,
            scores=scores,
        )
        logger.info(
            "Retrieved %d chunks (scores: %s)", len(documents), scores
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────
    def _init_vectorstore(self, force_reload: bool) -> None:
        """Load or create the ChromaDB vector store."""
        persist_dir = str(CHROMA_PERSIST_DIR)

        # Reuse persisted store if it exists and force_reload is False
        if not force_reload and Path(persist_dir).exists():
            try:
                self._vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self._embeddings,
                    collection_name="support_kb",
                )
                # Quick sanity check — make sure the collection has documents
                count = self._vectorstore._collection.count()
                if count > 0:
                    logger.info(
                        "Loaded existing ChromaDB store (%d chunks)", count
                    )
                    return
                logger.info("Persisted store is empty — rebuilding…")
            except Exception:
                logger.warning("Could not load persisted store — rebuilding…")

        self._build_vectorstore(persist_dir)

    def _build_vectorstore(self, persist_dir: str) -> None:
        """Load documents, split, embed, and persist."""
        kb_dir = str(KNOWLEDGE_BASE_DIR)

        if not Path(kb_dir).exists() or not any(Path(kb_dir).iterdir()):
            logger.warning(
                "Knowledge-base directory '%s' is empty or missing", kb_dir
            )
            return

        logger.info("Loading documents from %s …", kb_dir)
        loader = DirectoryLoader(
            kb_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        raw_docs = loader.load()
        logger.info("Loaded %d raw documents", len(raw_docs))

        if not raw_docs:
            logger.warning("No documents found — vector store will be empty")
            return

        chunks = self._splitter.split_documents(raw_docs)
        logger.info("Split into %d chunks (size=%d, overlap=%d)",
                     len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

        logger.info("Creating embeddings and persisting to ChromaDB …")
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=persist_dir,
            collection_name="support_kb",
        )
        logger.info("ChromaDB vector store created and persisted ✓")

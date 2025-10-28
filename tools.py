"""
Tools module for the Nutritionist Chatbot system.

This module provides two main tools:
1. web_search_tool: Performs web searches using Tavily and Wikipedia
2. rag_tool: Retrieves and ranks relevant documents from a Chroma vector database

Both tools are designed to be used as standalone functions that can be called
from LangGraph nodes or other parts of the system.
"""

from __future__ import annotations
import os
import re
import logging
import statistics as stats
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# LangChain / LangGraph components for document processing and LLM interactions
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# LLM and search providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader

# Vector database and embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Optional reranking service for improving search results
import cohere

# -----------------------------------------------------------------------------
# Logging / Environment / Shared Instances
# -----------------------------------------------------------------------------

# Set up logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# Load environment variables from .env file
load_dotenv()

# Shared LLM instance used for small structured subtasks like query generation and agent checks
# Using Gemini 2.0 Flash Experimental for fast, reliable responses
_LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

# Shared Tavily search instance with default max results
_TAVILY = TavilySearch(max_results=3)

# -----------------------------------------------------------------------------
# Helper Functions and Pydantic Schemas
# -----------------------------------------------------------------------------

def _clean_full(text: str) -> str:
    """
    Clean and normalize text by collapsing multiple whitespace characters into single spaces.
    
    Args:
        text: Input text to clean (can be None)
        
    Returns:
        Cleaned text with normalized whitespace
    """
    return re.sub(r"\s+", " ", (text or "")).strip()

def _truncate_log_content(content, max_words=12) -> str:
    """
    Truncate content for logging purposes to keep logs readable.
    
    Args:
        content: Content to truncate (any type)
        max_words: Maximum number of words to keep
        
    Returns:
        Truncated string representation of content
    """
    try:
        s = str(content)
    except Exception:
        return "<unprintable>"
    words = s.split()
    return s if len(words) <= max_words else " ".join(words[:max_words]) + "..."

class _SearchQuery(BaseModel):
    """
    Pydantic model for structured search query generation.
    Used by LLM to generate optimized search queries for different search engines.
    """
    search_query: str = Field(..., description="Search query for retrieval.")

class _AgentCheckResult(BaseModel):
    """
    Pydantic model for agent-based relevance checking results.
    Used to evaluate whether retrieved documents are relevant to the user's question.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="0..1 relevance score")
    contains_direct_answer: bool = Field(..., description="True if chunk alone likely answers")
    rationale: str = Field(..., description="1–2 short reasons for the score")

# -----------------------------------------------------------------------------
# WEB SEARCH TOOL (Tavily + Wikipedia)
# -----------------------------------------------------------------------------

def web_search_tool(
    question: str,
    *,
    max_results: int = 3,
    wiki_max_docs: int = 2,
) -> Dict[str, List[str]]:
    """
    Perform comprehensive web search using Tavily and Wikipedia.
    
    This tool implements a two-phase search strategy:
    1. Tavily web search for current/recent information
    2. Wikipedia search for authoritative reference material
    
    Both phases use LLM-generated optimized search queries to improve results.
    
    Args:
        question: The user's question to search for
        max_results: Maximum number of results from Tavily search
        wiki_max_docs: Maximum number of Wikipedia documents to load
        
    Returns:
        Dictionary with key "web_search_results" containing a list of formatted
        document strings. Each string is wrapped in <Document> tags with source
        information for easy parsing by the answer generation system.
        
    Example:
        >>> result = web_search_tool("What is vitamin B12?")
        >>> print(result["web_search_results"][0])
        <Document source="Tavily Web Search"/>
        Vitamin B12 is a water-soluble vitamin...
        </Document>
    """
    results: List[str] = []

    # Phase 1: Tavily Web Search
    # ---------------------------
    # Use LLM to generate an optimized search query for web search
    try:
        # Create a structured output generator for search queries
        qgen = _LLM.with_structured_output(_SearchQuery)
        
        # Generate optimized search query using LLM
        tavily_q = qgen.invoke([
            SystemMessage(content="Write a crisp web search query."),
            HumanMessage(content=question),
        ])
        logger.info(f"[web_tool] Tavily query: {tavily_q.search_query!r}")

        # Perform the actual web search using Tavily
        search = TavilySearch(max_results=max_results)  # Allow per-call override
        raw = search.invoke(tavily_q.search_query)
        
        # Format the results based on the response type
        if isinstance(raw, str) and raw.strip():
            # Single string response - wrap in Document tags
            formatted = f'<Document source="Tavily Web Search"/>\n{raw}\n</Document>'
            results.append(formatted)
        elif isinstance(raw, list) and raw:
            # List of search results - format each with URL and content
            formatted = "\n\n---\n\n".join(
                [f'<Document href="{d.get("url","N/A")}"/>\n{d.get("content", str(d))}\n</Document>' for d in raw]
            )
            results.append(formatted)
        elif raw:
            # Fallback for any other non-empty response
            formatted = f'<Document source="Tavily Web Search"/>\n{str(raw)}\n</Document>'
            results.append(formatted)
            
        logger.info(f"[web_tool] Tavily formatted len={len(results[-1]) if results else 0}")
    except Exception:
        logger.exception("[web_tool] Tavily phase failed")

    # Phase 2: Wikipedia Search
    # --------------------------
    # Use LLM to generate an optimized search query for Wikipedia
    try:
        # Create a new structured output generator for Wikipedia queries
        qgen = _LLM.with_structured_output(_SearchQuery)
        
        # Generate optimized Wikipedia search query using LLM
        wiki_q = qgen.invoke([
            SystemMessage(content="Write a crisp Wikipedia search query."),
            HumanMessage(content=question),
        ])
        logger.info(f"[web_tool] Wikipedia query: {wiki_q.search_query!r}")

        # Load Wikipedia documents using the generated query
        docs = WikipediaLoader(query=wiki_q.search_query, load_max_docs=wiki_max_docs).load()
        
        if docs:
            # Format Wikipedia results with source and page information
            formatted = "\n\n---\n\n".join(
                [
                    f'<Document source="{d.metadata.get("source")}" page="{d.metadata.get("page","")}"/>\n'
                    f'{d.page_content}\n</Document>'
                    for d in docs
                ]
            )
            results.append(formatted)
            logger.info(f"[web_tool] Wikipedia formatted len={len(formatted)}")
    except Exception:
        logger.exception("[web_tool] Wikipedia phase failed")

    # Return all search results in the expected format
    return {"web_search_results": results}

# -----------------------------------------------------------------------------
# RAG TOOL (Chroma → Cohere ReRank → Optional Agent-Check → Fuse/Prune → Format)
# -----------------------------------------------------------------------------

def rag_tool(
    question: str,
    *,
    f: int = 3,
    index_dir: str = "chroma-index",
    collection_name: str = "nutrition_knowledge_base",
) -> Dict[str, Any]:
    """
    Retrieve, rank, and format relevant documents from a Chroma vector database.
    
    This tool implements a sophisticated RAG (Retrieval-Augmented Generation) pipeline:
    1. Retrieve candidate documents using Chroma MMR (Maximal Marginal Relevance)
    2. Re-rank results using Cohere's reranking service for better relevance
    3. Optionally run agent-based relevance checks for quality control
    4. Fuse scores and apply diversity-aware selection
    5. Format results for consumption by the answer generation system
    
    Args:
        question: The user's question to search for
        f: Number of final documents to return (chunk budget)
        index_dir: Directory containing the Chroma vector database
        collection_name: Name of the Chroma collection to search
        
    Returns:
        Dictionary containing:
        - "RAG_results": List of formatted document strings with source/page info
        - "RAG_meta": List of metadata dictionaries for each result
        - "meta": Debug information and telemetry data
        
    Example:
        >>> result = rag_tool("What is vitamin B12?", f=3)
        >>> print(result["RAG_results"][0])
        Human-Nutrition-2020-Edition/page 45: Vitamin B12 is a water-soluble vitamin...
    """
    # Validate and initialize parameters
    f = max(1, int(f))  # Ensure f is at least 1
    meta: Dict[str, Any] = {"f": f}  # Track metadata for debugging

    # Phase 1: Document Retrieval using Chroma MMR
    # ----------------------------------------------
    # Use Maximal Marginal Relevance to balance relevance and diversity
    try:
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vs = Chroma(persist_directory=index_dir, collection_name=collection_name, embedding_function=embeddings)

        # Calculate retrieval parameters with headroom for better results
        # More candidates retrieved initially, then filtered down to f final results
        K_MULT, K_PAD, K_CAP = 6, 8, 120  # Multiplier, padding, and cap for k
        k = min(K_CAP, max(K_MULT * f, f + K_PAD))  # Number of final results to return
        fetch_k = max(120, int(4 * k))  # Number of candidates to fetch initially
        
        # Configure MMR retriever with diversity parameter
        retriever = vs.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": int(k), "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        
        # Retrieve candidate documents
        candidates: List[Document] = retriever.invoke(question)
        meta.update({"k": k, "fetch_k": fetch_k, "candidates_n": len(candidates)})
        logger.info(f"[rag_tool] Retrieved {len(candidates)} candidates (k={k}, fetch_k={fetch_k})")
    except Exception:
        logger.exception("[rag_tool] Retrieval failed")
        return {"RAG_results": [], "RAG_meta": [], "meta": {"error": "retrieval_failed"}}

    # Phase 2: Re-ranking using Cohere
    # --------------------------------
    # Use Cohere's reranking service to improve relevance scoring
    def _rerank(docs: List[Document], top_n: int) -> List[Document]:
        """
        Re-rank documents using Cohere's reranking service.
        
        Args:
            docs: List of documents to re-rank
            top_n: Number of top documents to return (unused, computed internally)
            
        Returns:
            Re-ranked list of documents with updated metadata including rerank scores
        """
        if not docs:
            return []
            
        # Check for Cohere API key
        api_key = os.getenv("COHERE_API_KEY")
        
        # Calculate how many documents to re-rank (with reasonable limits)
        r_top = min(len(docs), max(int(2.0 * f), f + 2), 24)
        meta["r_top"] = r_top
        
        # Fallback to original order if no API key
        if not api_key:
            logger.warning("[rag_tool] COHERE_API_KEY not set; using retriever order")
            return docs[:r_top]
            
        try:
            # Initialize Cohere client and extract document texts
            client = cohere.Client(api_key)
            txts = [d.page_content or "" for d in docs]
            
            # Perform reranking using Cohere's English model
            resp = client.rerank(model="rerank-english-v3.0", query=question, documents=txts, top_n=r_top)
            
            # Reorder documents based on reranking results
            ordered: List[Document] = []
            for r in resp.results:
                d = docs[r.index]  # Get original document by index
                md = d.metadata or {}
                
                # Store the rerank score in metadata
                try:
                    md["rerank_score"] = float(r.relevance_score)
                except Exception:
                    md["rerank_score"] = None
                    
                d.metadata = md
                ordered.append(d)
            return ordered
            
        except Exception:
            logger.exception("[rag_tool] Cohere rerank failed; using retriever order")
            return docs[:r_top]

    # Apply reranking to candidates
    reranked: List[Document] = _rerank(candidates, top_n=f)  # top_n ignored; r_top computed internally
    meta["reranked_n"] = len(reranked)

    # Phase 3: Agent-Based Quality Control Gate
    # -----------------------------------------
    # Determine whether to run expensive agent-based relevance checks
    def _should_call_agent(docs: List[Document]) -> Dict[str, Any]:
        """
        Determine whether to run expensive agent-based relevance checks.
        
        Uses statistical analysis of rerank scores to decide:
        - Skip if we don't have enough candidates
        - Skip if rerank scores are already very high and well-separated
        - Run agent checks if scores are ambiguous or low
        
        Args:
            docs: List of reranked documents
            
        Returns:
            Dictionary with skip decision and statistical metrics
        """
        n = len(docs)
        MIN_HEADROOM = 2  # Need at least this many extra candidates
        
        # Skip if we don't have enough candidates for meaningful selection
        if n < f + MIN_HEADROOM:
            return {"skip": True, "top_f_mean_z": None, "gap_sigma": None, "n": n}
            
        # Extract rerank scores from document metadata
        scores: List[Optional[float]] = []
        for d in docs:
            s = d.metadata.get("rerank_score")
            try:
                scores.append(float(s) if s is not None else None)
            except Exception:
                scores.append(None)
                
        # Skip if any scores are missing (can't do statistical analysis)
        if any(s is None for s in scores):
            return {"skip": False, "top_f_mean_z": None, "gap_sigma": None, "n": n}
            
        # Calculate statistical metrics for decision making
        mean = stats.mean(scores)
        stdev = stats.pstdev(scores) or 1e-6  # Avoid division by zero
        z = [(s - mean) / stdev for s in scores]  # Z-scores for normalization
        
        # Calculate metrics for top f documents
        top_f_mean_z = stats.mean(z[:f])  # Average z-score of top f
        gap_sigma = z[f-1] - z[f]  # Gap between f-th and (f+1)-th document
        
        # Decision thresholds
        SKIP_IF_TOPF_MEAN_Z = 0.9  # Skip if top documents are very high scoring
        SKIP_IF_GAP_SIGMA   = 0.5  # Skip if there's a clear gap after f documents
        
        # Skip agent checks if scores are already very good and well-separated
        skip = (top_f_mean_z >= SKIP_IF_TOPF_MEAN_Z) and (gap_sigma >= SKIP_IF_GAP_SIGMA)
        return {"skip": skip, "top_f_mean_z": top_f_mean_z, "gap_sigma": gap_sigma, "n": n}

    # Make decision about running agent checks
    conf = _should_call_agent(reranked)
    meta["confidence"] = conf
    logger.info(f"[rag_tool] confidence: {conf}")

    # Phase 4: Optional Agent-Based Relevance Checking
    # ------------------------------------------------
    # Use LLM to evaluate each document's relevance to the question
    flags: List[Dict[str, Any]] = []
    if not conf["skip"]:
        # Initialize structured output generator for agent checks
        checker = _LLM.with_structured_output(_AgentCheckResult)
        sys = SystemMessage(content=(
            "You are a fast, strict relevance scorer. "
            "Score how useful the CHUNK is for answering the QUESTION. "
            "Return score in [0,1], a boolean for direct answer, and a brief rationale."
        ))
        
        # Check each document individually
        for idx, d in enumerate(reranked):
            try:
                # Create human message with question and document content
                human = HumanMessage(content=f"QUESTION:\n{question}\n\nCHUNK:\n{d.page_content}")
                res: _AgentCheckResult = checker.invoke([sys, human])
                
                # Extract metadata and store agent check results
                md = d.metadata or {}
                flags.append({
                    "doc_index": idx,
                    "score_agent": float(res.score),
                    "contains_direct_answer": bool(res.contains_direct_answer),
                    "rationale": (res.rationale or "")[:300],  # Truncate rationale
                    "source": md.get("source"),
                    "page": md.get("page"),
                    "chunk_id": md.get("chunk_id"),
                    "rerank_score": float(md.get("rerank_score")) if isinstance(md.get("rerank_score"), (int, float)) else None,
                })
                logger.info("[rag_tool] agent_check idx=%d score=%.2f direct=%s",
                            idx, flags[-1]["score_agent"], flags[-1]["contains_direct_answer"])
            except Exception:
                # Fallback for failed agent checks
                logger.exception("[rag_tool] agent_check failed for idx=%d", idx)
                md = d.metadata or {}
                flags.append({
                    "doc_index": idx,
                    "score_agent": 0.5,  # Default neutral score
                    "contains_direct_answer": False,
                    "rationale": "checker_error",
                    "source": md.get("source"),
                    "page": md.get("page"),
                    "chunk_id": md.get("chunk_id"),
                    "rerank_score": md.get("rerank_score"),
                })
    meta["flags_n"] = len(flags)

    # Phase 5: Score Fusion and Diversity-Aware Selection
    # ---------------------------------------------------
    # Combine rerank scores with agent scores and apply diversity penalties
    def _fuse_prune(docs: List[Document], flags: List[Dict[str, Any]], f: int) -> List[Document]:
        """
        Fuse rerank and agent scores, then apply diversity-aware selection.
        
        This function:
        1. Normalizes rerank scores to [0,1] range
        2. Combines rerank scores with agent scores using weighted fusion
        3. Applies diversity penalties to avoid selecting similar documents
        4. Selects the best f documents using greedy selection
        
        Args:
            docs: List of documents to select from
            flags: Agent check results for each document
            f: Number of documents to select
            
        Returns:
            List of f selected documents, sorted by fused score
        """
        n = len(docs)
        if n == 0:
            return []

        # Create mapping from document index to agent check results
        agent_by_idx = {rec.get("doc_index"): rec for rec in flags if isinstance(rec.get("doc_index"), int)}

        # Normalize rerank scores to [0,1] range for consistent fusion
        raw = []
        for d in docs:
            s = d.metadata.get("rerank_score")
            try:
                raw.append(float(s) if s is not None else None)
            except (TypeError, ValueError):
                raw.append(None)
                
        valid = [x for x in raw if x is not None]
        if not valid:
            norm = [0.5] * n  # Default neutral score if no valid scores
        else:
            lo, hi = min(valid), max(valid)
            # Min-max normalization: (x - min) / (max - min)
            norm = [0.5 if hi == lo else (((x - lo) / (hi - lo)) if x is not None else 0.0) for x in raw]

        # Score Fusion: Combine rerank scores with agent scores
        # -----------------------------------------------------
        # Weighted combination: 70% rerank + 30% agent + bonus for direct answers
        ALPHA, BETA, BONUS_DIRECT, ABS_MIN = 0.7, 0.3, 0.05, 0.25
        
        for i, d in enumerate(docs):
            rerank_s = norm[i]
            agent_rec = agent_by_idx.get(i)
            
            if agent_rec:
                # Combine rerank and agent scores with bonus for direct answers
                agent_s = float(agent_rec.get("score_agent", 0.5))
                contains = bool(agent_rec.get("contains_direct_answer", False))
                fused = ALPHA * rerank_s + BETA * agent_s + (BONUS_DIRECT if contains else 0.0)
            else:
                # Use only rerank score if no agent check available
                fused = rerank_s
                
            # Store fused score in document metadata
            d.metadata = dict(d.metadata or {})
            d.metadata["fused_score"] = max(0.0, min(1.0, fused))  # Clamp to [0,1]

        # Diversity-Aware Greedy Selection
        # ---------------------------------
        # Apply penalties for selecting documents from same source/page to encourage diversity
        PAGE_PENALTY, SRC_PENALTY = 0.15, 0.05
        
        # Sort documents by fused score (descending)
        order = sorted(range(n), key=lambda i: docs[i].metadata.get("fused_score", 0.0), reverse=True)
        selected: List[int] = []
        seen_page, seen_src = {}, {}  # Track diversity
        
        def adjusted(i: int) -> float:
            """Calculate adjusted score with diversity penalties."""
            d = docs[i]
            src = d.metadata.get("source", "unknown_source")
            page = d.metadata.get("page")
            base = d.metadata.get("fused_score", 0.0)
            
            # Apply penalties for repeated sources/pages
            return base - PAGE_PENALTY * seen_page.get((src, page), 0) - SRC_PENALTY * seen_src.get(src, 0)

        # Greedy selection with diversity awareness
        remaining = set(order)
        while len(selected) < f and remaining:
            # Find document with highest adjusted score
            best_i, best_score = None, float("-inf")
            for i in remaining:
                sc = adjusted(i)
                if sc > best_score:
                    best_i, best_score = i, sc
                    
            # Select the best document and update diversity tracking
            selected.append(best_i)
            remaining.remove(best_i)
            d = docs[best_i]
            src = d.metadata.get("source", "unknown_source")
            page = d.metadata.get("page")
            seen_page[(src, page)] = seen_page.get((src, page), 0) + 1
            seen_src[src] = seen_src.get(src, 0) + 1

        # Final Quality Filter and Sorting
        # ---------------------------------
        # Ensure we have enough high-quality documents
        picked = [docs[i] for i in selected]
        strong = [d for d in picked if (d.metadata.get("fused_score") or 0.0) >= ABS_MIN]
        
        # Fill remaining slots with best available documents if needed
        if len(strong) < f:
            pool = sorted(list(remaining), key=lambda i: docs[i].metadata.get("fused_score", 0.0), reverse=True)
            for i in pool[:(f - len(strong))]:
                strong.append(docs[i])
                
        # Sort final results by fused score (descending)
        strong.sort(key=lambda d: d.metadata.get("fused_score") or 0.0, reverse=True)
        return strong

    # Apply fusion and selection to get final documents
    final_docs = _fuse_prune(reranked, flags, f)
    meta["final_n"] = len(final_docs)

    # Phase 6: Format Results for Answer Generation
    # ---------------------------------------------
    # Convert documents to formatted strings with source information
    rag_results: List[str] = []
    rag_meta: List[Dict[str, Any]] = []
    
    for d in final_docs:
        # Extract metadata for formatting
        src = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page")
        chunk_id = d.metadata.get("chunk_id")
        score = d.metadata.get("rerank_score", d.metadata.get("relevance_score"))
        
        # Create formatted string with source/page information
        prefix = f"{src}" if page is None else f"{src}/page {page}"
        line = f"{prefix}: {_clean_full(d.page_content)}"
        rag_results.append(line)
        
        # Store metadata for debugging and analysis
        rag_meta.append({
            "source": src, 
            "page": page, 
            "chunk_id": chunk_id,
            "rerank_score": float(score) if isinstance(score, (int, float)) else None,
        })
        
        logger.info("[rag_tool] RAG line preview: %s", _truncate_log_content(line))

    # Return formatted results with metadata
    return {"RAG_results": rag_results, "RAG_meta": rag_meta, "meta": meta}

__all__ = ["web_search_tool", "rag_tool"]
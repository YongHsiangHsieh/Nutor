from langchain_core.documents.base import Document
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
import operator
from typing import List, Optional, Annotated, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
# Web search tool
from langchain_tavily import TavilySearch
# Wikipedia search tool
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv
import os
import logging
import uuid
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# Reranker
import re
# Optional Cohere reranker
import cohere
# Optional: use HF embeddings if you indexed with Sentence Transformers
# from langchain_community.embeddings import HuggingFaceEmbeddings
import statistics as stats

# Configure logging to only print to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def truncate_log_content(content, max_words=10):
    """Helper function to truncate long content in logs to avoid overwhelming output."""
    if not content:
        return str(content)
    
    content_str = str(content)
    words = content_str.split()
    if len(words) <= max_words:
        return content_str
    
    truncated_words = words[:max_words]
    return ' '.join(truncated_words) + '...'

# Load env FIRST before initializing any tools that need API keys
logger.info("Loading environment variables...")
load_dotenv()
logger.info("Environment variables loaded successfully")

# Now initialize tools that need API keys
logger.info("Initializing TavilySearch tool (max_results=3)...")
tavily_search = TavilySearch(max_results=3)
logger.info("TavilySearch tool initialized successfully")

logger.info("Initializing ChatGoogleGenerativeAI model (gemini-2.0-flash-exp)...")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
logger.info("LLM model initialized successfully") 

class OverallState(TypedDict):
    question: str
    answer: str
    is_relevant: bool
    web_search_results: list[str]
    RAG_results: list[str]
    f: int  # single control knob: how many chunks to pass to generate_answer

# Question checking
check_instructions = SystemMessage(content=f"""You are a helpful assistant specialized in checking if a question is relevant to nutrition, fitness, health, and wellness.""")

class CheckQuestion(BaseModel):
    is_relevant: bool = Field(None, description="Whether the question is relevant to nutrition, fitness, health, and wellness.")

def check_question(state: OverallState):

    """ Check if the question is relevant to nutrition, fitness, health, and wellness """
    
    logger.info("="*50)
    logger.info("ENTERING check_question function")
    logger.info(f"State received: {truncate_log_content(state)}")
    
    # Check question
    logger.info("Creating structured LLM with CheckQuestion schema...")
    structured_llm = llm.with_structured_output(CheckQuestion)
    
    logger.info("Invoking LLM to check question relevance...")
    messages = [check_instructions, HumanMessage(content=state['question'])]
    is_relevant = structured_llm.invoke(messages)
    
    logger.info(f"Question relevance result: {is_relevant.is_relevant}")
    result = {"is_relevant": is_relevant.is_relevant}
    logger.info(f"EXITING check_question with result: {truncate_log_content(result)}")
    logger.info("="*50)
    
    return result


# Search Sub graph
class SearchState(TypedDict):
    web_search_results: Annotated[list[str], operator.add]
    question: str

class SearchOut(TypedDict):
    web_search_results: Annotated[list[str], operator.add]  # only thing this branch writes

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

# Search query writing
search_instructions = SystemMessage(content=f"""You are a helpful assistant specialized in writing search queries for web and wikipedia search.""")

def search_web(state: SearchState):
    question = state["question"]
    logger.info(f"Question: {question}")
    """ Retrieve docs from web search """
    
    logger.info("="*50)
    logger.info("ENTERING search_web function")
    logger.info(f"State received: {truncate_log_content(state)}")

    # Search query
    logger.info("Creating structured LLM with SearchQuery schema...")
    structured_llm = llm.with_structured_output(SearchQuery)
    
    logger.info("Invoking LLM to generate web search query...")
    messages = [search_instructions, HumanMessage(content=question)]
    search_query = structured_llm.invoke(messages)
    logger.info(f"Generated search query: {search_query.search_query}")
    
    # Search
    logger.info(f"Performing Tavily web search for: {search_query.search_query}")
    search_result = tavily_search.invoke(search_query.search_query)
    logger.info(f"Tavily search completed. Result type: {type(search_result)}")
    logger.info(f"Search result preview: {truncate_log_content(search_result)}")
    
     # Format
    logger.info("Formatting web search results...")
    # TavilySearch returns a string directly, not a list of dicts
    if isinstance(search_result, str):
        formatted_search_docs = f'<Document source="Tavily Web Search"/>\n{search_result}\n</Document>'
        if not search_result.strip():
            logger.info("Tavily search returned empty result; returning empty string.")
            return {"web_search_results": []}
    elif isinstance(search_result, list):
        if not search_result:  # nothing found
            logger.info("Tavily search returned empty result; returning empty string.")
            return {"web_search_results": []}
        # Handle if it returns a list
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc.get("url", "N/A")}"/>\n{doc.get("content", str(doc))}\n</Document>'
                for doc in search_result
            ]
        )
    else:
        if not search_result:
            logger.info("Tavily search returned empty result; returning empty string.")
            return {"web_search_results": []}
        # Fallback
        formatted_search_docs = f'<Document source="Tavily Web Search"/>\n{str(search_result)}\n</Document>'
    
    logger.info(f"Formatted web search results (length: {len(formatted_search_docs)} chars)")
    
    result = {"web_search_results": [formatted_search_docs]}
    logger.info(f"EXITING search_web with {len(result['web_search_results'])} result(s)")
    logger.info("="*50)

    return result 

def search_wikipedia(state: SearchState):
    
    """ Retrieve docs from wikipedia """
    
    logger.info("="*50)
    logger.info("ENTERING search_wikipedia function")
    logger.info(f"State received: {truncate_log_content(state)}")
    question = state["question"]
    logger.info(f"Question: {question}")
    # Search query
    logger.info("Creating structured LLM with SearchQuery schema...")
    structured_llm = llm.with_structured_output(SearchQuery)
    
    logger.info("Invoking LLM to generate Wikipedia search query...")
    messages = [search_instructions, HumanMessage(content=question)]
    search_query = structured_llm.invoke(messages)
    logger.info(f"Generated search query: {search_query.search_query}")
    
    # Search
    logger.info(f"Loading Wikipedia documents for: {search_query.search_query} (max_docs=2)")
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()
    if not search_docs:               # <— guard
        logger.info("Wikipedia search returned empty result; returning empty string.")
        return {"web_search_results": []}
        
    logger.info(f"Found {len(search_docs)} Wikipedia documents")

     # Format
    logger.info("Formatting Wikipedia search results...")
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    logger.info(f"Formatted Wikipedia results (length: {len(formatted_search_docs)} chars): {truncate_log_content(formatted_search_docs)}")
    
    result = {"web_search_results": [formatted_search_docs]}
    logger.info(f"EXITING search_wikipedia with {len(result['web_search_results'])} result(s)")
    logger.info("="*50)

    return result 

class RAGState(TypedDict):
    # Inputs
    question: str
    f: int  # single control knob (final number of chunks to feed generate_answer)

    # Internals
    candidates: List[Document]
    reranked: List[Document]               # after rerank (Cohere)
    skip_checker: bool                     # set by gate
    confidence: Dict[str, Any]
    flags: Annotated[List[Dict], operator.add]   # LLM checker outputs (one per doc) 
    RAG_results: Annotated[List[str], operator.add]
    RAG_meta: Annotated[List[Dict], operator.add]

class RAGOut(TypedDict):
    RAG_results: Annotated[List[str], operator.add]
    RAG_meta:    Annotated[List[Dict], operator.add]

def rag_retrieval(state: RAGState):
    """Retrieve candidate chunks from a persisted Chroma collection. Recall size derived from f."""
    logger.info("=" * 50)
    logger.info("ENTERING rag_retrieval function")
    logger.info(f"State received: {truncate_log_content(state)}")

    question = state["question"]
    f = max(1, int(state.get("f", 3)))  # single control knob

    # Derive recall size k from f with headroom and caps
    K_MULT = 6
    K_PAD  = 8
    K_CAP  = 120
    k = min(K_CAP, max(K_MULT * f, f + K_PAD))

    try:
        embeddings = OpenAIEmbeddings()  # must match ingestion
        index_dir = "chroma-index"
        collection_name = "nutrition_knowledge_base"
        logger.info(f"Opening Chroma collection from: {index_dir} (collection='{collection_name}')")
        vs = Chroma(
            embedding_function=embeddings,
            persist_directory=index_dir,
            collection_name=collection_name,
        )

        # Wider candidate pool before MMR trims redundancy
        fetch_k = max(120, int(4 * k))
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": int(k), "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        logger.info(f"Retrieving (MMR) top-{k} from fetch_k={fetch_k} for question: {question!r}")
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} candidate docs")

        result = {"candidates": docs}
        logger.info(f"EXITING rag_retrieval with result count: {len(docs)}")
        logger.info("=" * 50)
        return result

    except Exception as e:
        logger.exception("rag_retrieval (Chroma) failed")
        return {"candidates": []}
    

def rag_rerank(state: RAGState):
    """Rerank candidates to a top list derived from f using Cohere SDK. No formatting."""
    logger.info("=" * 50)
    logger.info("ENTERING rag_rerank function")
    logger.info(f"State received (keys): {list(state.keys())}")

    question = state["question"]
    f = max(1, int(state.get("f", 3)))
    candidates: List[Document] = state.get("candidates", []) or []

    if not candidates:
        logger.info("No candidates provided; returning empty reranked list.")
        return {"reranked": []}

    # Derive r_top from f with headroom and caps (must be > f to give pruning/agent room)
    R_MULT = 2.0
    R_PAD  = 2
    R_CAP  = 24
    r_top = min(len(candidates), R_CAP, max(int(R_MULT * f), f + R_PAD))

    try:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not set; falling back to retriever order.")
            return {"reranked": candidates[:r_top]}

        client = cohere.Client(api_key)
        docs_text = [d.page_content or "" for d in candidates]

        logger.info(f"Using Cohere ReRank (SDK) model=rerank-english-v3.0; docs={len(docs_text)}, top_n={r_top}")
        resp = client.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=docs_text,
            top_n=r_top,
        )

        ordered: List[Document] = []
        for r in resp.results:
            d = candidates[r.index]
            try:
                d.metadata["rerank_score"] = float(r.relevance_score)
            except Exception:
                d.metadata["rerank_score"] = None
            ordered.append(d)

        logger.info(f"Reranked {len(candidates)} → top-{len(ordered)} via Cohere SDK")
        return {"reranked": ordered}

    except Exception:
        logger.exception("rag_rerank (Cohere SDK) failed; returning fallback order.")
        return {"reranked": candidates[:r_top]}

def should_call_agent(state: RAGState):
    """
    Decide whether to run the per-chunk agent checker using f as the target final count.

    Logic:
    - Require headroom: at least f + MIN_HEADROOM reranked docs; otherwise skip to save cost.
    - If headroom exists, compute z-scores over rerank scores.
      If top-f is strong AND there's a clear gap after f, skip; else call agent.
    """
    logger.info("=" * 50)
    logger.info("ENTERING should_call_agent")

    docs = state.get("reranked", []) or []
    f = max(1, int(state.get("f", 3)))
    n = len(docs)

    MIN_HEADROOM = 2
    if n < f + MIN_HEADROOM:
        logger.info(f"Not enough headroom (n={n}, f={f}); skipping agent checker.")
        return {
            "skip_checker": True,
            "confidence": {"f": f, "n": n, "top_f_mean_z": None, "gap_sigma": None}
        }

    scores = []
    for d in docs:
        s = d.metadata.get("rerank_score")
        try:
            s = float(s) if s is not None else None
        except Exception:
            s = None
        scores.append(s)

    if any(s is None for s in scores):
        logger.info("Some rerank scores missing; calling agent checker for safety.")
        return {
            "skip_checker": False,
            "confidence": {"f": f, "n": n, "top_f_mean_z": None, "gap_sigma": None}
        }

    mean = stats.mean(scores)
    stdev = stats.pstdev(scores) or 1e-6
    z = [(s - mean) / stdev for s in scores]

    top_f_mean_z = stats.mean(z[:f])
    gap_sigma = z[f-1] - z[f]  # separation between f and f+1

    SKIP_IF_TOPF_MEAN_Z = 0.9
    SKIP_IF_GAP_SIGMA   = 0.5
    skip = (top_f_mean_z >= SKIP_IF_TOPF_MEAN_Z) and (gap_sigma >= SKIP_IF_GAP_SIGMA)

    logger.info(
        f"Confidence signals: n={n}, f={f}, top_f_mean_z={top_f_mean_z:.2f}, gap_sigma={gap_sigma:.2f} "
        f"→ skip_checker={skip}"
    )

    return {
        "skip_checker": skip,
        "confidence": {
            "f": f,
            "n": n,
            "top_f_mean_z": top_f_mean_z,
            "gap_sigma": gap_sigma,
        },
    }

def route_after_should_call_agent(state: RAGState) -> str:
    skip = bool(state.get("skip_checker", False))
    logger.info(f"route_after_should_call_agent: skip_checker={skip}")
    return "rag_fuse_prune" if skip else "agent_entry"

def agent_entry(state: RAGState):
    """
    It's a fan-out node that schedules one checker task per reranked doc.
    """
    logger.info("agent_entry: entering")
    return {}

# Agent check result
class AgentCheckResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="0..1 relevance score")
    contains_direct_answer: bool = Field(..., description="True if this chunk alone likely answers the question")
    rationale: str = Field(..., description="1–2 short reasons for the score")


def to_agent_check(state: RAGState):
    """
    Fan out: schedule one checker task per reranked doc.
    We send the *doc payload* so the worker is stateless and robust.
    """
    docs = state.get("reranked", []) or []
    if not docs:
        logger.info("to_agent_check: no docs; nothing to send.")
        return []

    q = state.get("question", "")
    sends = []
    for idx, d in enumerate(docs):
        md = d.metadata or {}
        sends.append(Send(
            "rag_agent_check",
            {
                "question": q,
                # Minimal self-contained payload for the worker
                "doc": {
                    "text": d.page_content or "",
                    "source": md.get("source"),
                    "page": md.get("page"),
                    "chunk_id": md.get("chunk_id"),
                    "rerank_score": md.get("rerank_score"),
                    "doc_index": idx,   # optional: handy for logs/fan-in mapping
                }
            }
        ))
    logger.info(f"to_agent_check: dispatched {len(sends)} checker tasks")
    return sends

# Rag agent check
def rag_agent_check(state: RAGState):
    """
    Per-doc worker: score a single chunk passed in `state["doc"]`.
    Stateless: does not rely on `reranked`.
    """
    try:
        payload = state.get("doc")
        if not payload or "text" not in payload:
            raise ValueError("rag_agent_check: missing 'doc' payload with 'text' field")

        question = state.get("question") or ""
        doc_text = payload["text"]
        src = payload.get("source", "unknown_source")
        page = payload.get("page")
        chunk_id = payload.get("chunk_id")
        cohere_score = payload.get("rerank_score")
        doc_idx = payload.get("doc_index")  # optional

        checker = llm.with_structured_output(AgentCheckResult)
        sys = SystemMessage(content=(
            "You are a fast and accurate relevance scorer. "
            "Score how useful the CHUNK is for answering the QUESTION. "
            "Return a score in [0,1], a boolean if it likely contains a direct answer, "
            "and a brief rationale (<= 2 lines). Be strict about relevance."
        ))
        human = HumanMessage(content=f"QUESTION:\n{question}\n\nCHUNK:\n{doc_text}")
        res: AgentCheckResult = checker.invoke([sys, human])

        rec = {
            "doc_index": doc_idx,
            "score_agent": float(res.score),
            "contains_direct_answer": bool(res.contains_direct_answer),
            "rationale": (res.rationale or "")[:300],
            "source": src,
            "page": page,
            "chunk_id": chunk_id,
            "rerank_score": float(cohere_score) if isinstance(cohere_score, (int, float)) else None,
        }
        logger.info(
            "rag_agent_check: idx=%s score_agent=%.2f direct=%s",
            str(doc_idx), rec["score_agent"], rec["contains_direct_answer"]
        )
        return {"flags": [rec]}

    except Exception:
        logger.exception("rag_agent_check failed; returning neutral record")
        return {"flags": [{
            "doc_index": state.get("doc", {}).get("doc_index"),
            "score_agent": 0.5,
            "contains_direct_answer": False,
            "rationale": "checker_error",
            "source": state.get("doc", {}).get("source"),
            "page": state.get("doc", {}).get("page"),
            "chunk_id": state.get("doc", {}).get("chunk_id"),
            "rerank_score": state.get("doc", {}).get("rerank_score"),
        }]}

# Agent gate
def agent_gate(state: RAGState):
    """
    Barrier: no-op. Ensures all rag_agent_check tasks completed before proceeding.
    """
    total = len(state.get("reranked", []) or [])
    got = len(state.get("flags", []) or [])
    logger.info(f"agent_gate: collected flags for {got}/{total} docs")
    return {}

# Rag fuse prune
def rag_fuse_prune(state: RAGState):
    """
    Fuse reranker scores with optional agent scores; apply soft diversity penalties;
    prune to final f; return the docs under 'reranked' for the formatter.
    """
    logger.info("=" * 50)
    logger.info("ENTERING rag_fuse_prune")

    docs: List[Document] = state.get("reranked", []) or []
    flags: List[Dict] = state.get("flags", []) or []
    logger.info(f"Length of flags: {len(flags)}")
    f = max(1, int(state.get("f", 3)))
    n = len(docs)

    if n == 0:
        logger.info("rag_fuse_prune: no docs; returning empty.")
        return {"reranked": []}

    # 1) Map agent flags by document index
    agent_by_idx: Dict[int, Dict] = {}
    for rec in flags:
        idx = rec.get("doc_index")
        if isinstance(idx, int) and 0 <= idx < n:
            agent_by_idx[idx] = rec

    # 2) Normalize rerank scores to [0,1]
    raw = []
    for d in docs:
        s = d.metadata.get("rerank_score")
        try:
            raw.append(float(s) if s is not None else None)
        except (TypeError, ValueError):
            raw.append(None)
    valid = [x for x in raw if x is not None]
    if not valid:
        norm = [0.5] * n
    else:
        lo, hi = min(valid), max(valid)
        if hi == lo:
            norm = [0.5 if x is not None else 0.0 for x in raw]
        else:
            norm = [((x - lo) / (hi - lo)) if x is not None else 0.0 for x in raw]

    # 3) Fuse rerank + agent
    ALPHA = 0.7
    BETA  = 0.3
    BONUS_DIRECT = 0.05
    ABS_MIN = 0.25

    fused = []
    for i, d in enumerate(docs):
        rerank_s = norm[i]
        agent_rec = agent_by_idx.get(i)
        if agent_rec is not None:
            agent_s = float(agent_rec.get("score_agent", 0.5))
            contains = bool(agent_rec.get("contains_direct_answer", False))
            score = ALPHA * rerank_s + BETA * agent_s + (BONUS_DIRECT if contains else 0.0)
        else:
            contains = False
            score = rerank_s
        score = max(0.0, min(1.0, score))
        md = d.metadata or {}
        md["agent_score"] = agent_rec.get("score_agent") if agent_rec is not None else None
        md["contains_direct_answer"] = contains if agent_rec is not None else False
        md["fused_score"] = score
        d.metadata = md
        fused.append((i, score))

    # 4) Diversity-aware greedy selection
    PAGE_PENALTY = 0.15
    SRC_PENALTY  = 0.05
    fused.sort(key=lambda x: x[1], reverse=True)

    selected: List[int] = []
    seen_per_page: Dict[tuple, int] = {}
    seen_per_source: Dict[str, int] = {}

    def adjusted_score(idx: int, base: float) -> float:
        d = docs[idx]
        src = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page")
        key_page = (src, page)
        page_count = seen_per_page.get(key_page, 0)
        src_count = seen_per_source.get(src, 0)
        return base - PAGE_PENALTY * page_count - SRC_PENALTY * src_count

    remaining = set(i for i, _ in fused)
    while len(selected) < f and remaining:
        best_idx = None
        best_adj = float("-inf")
        for i in remaining:
            base = docs[i].metadata.get("fused_score", 0.0)
            adj = adjusted_score(i, base)
            if adj > best_adj:
                best_adj, best_idx = adj, i
        selected.append(best_idx)
        remaining.remove(best_idx)
        d = docs[best_idx]
        src = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page")
        key_page = (src, page)
        seen_per_page[key_page] = seen_per_page.get(key_page, 0) + 1
        seen_per_source[src] = seen_per_source.get(src, 0) + 1

    picked = [docs[i] for i in selected]

    # 5) Prune by floor + backfill to exactly f
    strong = [d for d in picked if (d.metadata.get("fused_score") or 0.0) >= ABS_MIN]
    if len(strong) < f:
        pool = [docs[i] for i in remaining]
        pool.sort(key=lambda d: d.metadata.get("fused_score") or 0.0, reverse=True)
        take = f - len(strong)
        strong.extend(pool[:take])

    strong.sort(key=lambda d: d.metadata.get("fused_score") or 0.0, reverse=True)

    logger.info(
        "rag_fuse_prune: input=%d, flags=%d, output=%d, fused_top=[%s]",
        n, len(flags), len(strong),
        ", ".join(f"{(d.metadata.get('source') or 'src')}#p{d.metadata.get('page')}:{(d.metadata.get('fused_score') or 0):.2f}"
                  for d in strong[:min(5, len(strong))])
    )
    return {"reranked": strong}

# Format RAG results
def format_rag_results(state: RAGState):
    """
    Turn `reranked` Documents into:
      - RAG_results: List[str] -> "source/page X: <cleaned full text (no truncation)>"
      - RAG_meta:    List[Dict] -> {"source", "page", "chunk_id", "rerank_score"}
    NOTE: We CLEAN whitespace but DO NOT TRUNCATE, so the LLM sees the full chunk content.
    """
    logger.info("=" * 50)
    logger.info("ENTERING format_rag_results function")
    logger.info(f"State received (keys): {list(state.keys())}")

    docs: List[Document] = state.get("reranked", []) or []
    if not docs:
        logger.info("No reranked docs provided; returning empty outputs.")
        return {"RAG_results": [], "RAG_meta": []}

    def _clean_full(text: str) -> str:
        # Collapse any run of whitespace (spaces/tabs/newlines) to a single space, keep ALL words.
        # No truncation. Leading/trailing whitespace removed.
        return re.sub(r"\s+", " ", (text or "")).strip()

    results: List[str] = []
    meta: List[Dict] = []

    for d in docs:
        src = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page")  # may be None
        chunk_id = d.metadata.get("chunk_id")
        score = d.metadata.get("rerank_score", d.metadata.get("relevance_score"))

        prefix = f"{src}" if page is None else f"{src}/page {page}"
        # ✅ Use FULL content (no truncation), only cleaned:
        cleaned = _clean_full(d.page_content)
        line = f"{prefix}: {cleaned}"
        results.append(line)

        meta.append({
            "source": src,
            "page": page,
            "chunk_id": chunk_id,
            "rerank_score": float(score) if isinstance(score, (int, float)) else None,
        })

        # Optional small log preview (truncated to avoid overwhelming logs)
        logger.info(f"RAG line preview: {truncate_log_content(line)}")

    logger.info(f"Formatted {len(results)} RAG result lines (cleaned, no truncation)")
    logger.info("EXITING format_rag_results")
    logger.info("=" * 50)
    return {"RAG_results": results, "RAG_meta": meta}


# Generae answer
instructions = SystemMessage(content="""You are an experienced nutritionist who answers questions about nutrition, fitness, health, and wellness.

PRIORITY & SCOPE
- Use ONLY the information provided in the user message that follows (it contains the Question, RAG Results, and Web Results).
- PRIORITIZE RAG Results over Web Results. If there is any conflict, prefer RAG Results.
- If the provided context is insufficient to answer, reply exactly: "I don't know".

CITATION RULES (MANDATORY)
- Every factual claim must be followed immediately by an inline citation.
- For RAG Results lines formatted like "{source}/page {page}: ...", cite as: (RAG: {source}, p.{page}).
- For Web Results, cite as: (WEB: {source_or_domain}). If a full URL is present, extract and cite the domain.
- End with a "References" section listing each unique source you used (RAG and WEB), one per line, using the same labels.

OUTPUT FORMAT
1) Provide the answer first. Keep it concise, accurate, and grounded in the given context.
2) After the answer, add:
   References:
   - (RAG: <source>, p.<page>) — one per unique RAG source you used
   - (WEB: <source_or_domain>) — one per unique Web source you used

GUARDRAILS
- Do not invent or assume facts not present in the provided context.
- Do not quote or reference sources that are not in the provided context.
""")

class Answer(BaseModel):
    answer: str = Field(None, description="Answer to the question.")

def generate_answer(state: OverallState):
    """ Generate final answer based on RAG and web search results """
    
    logger.info("="*50)
    logger.info("ENTERING generate_answer function")
    logger.info(f"State received: {truncate_log_content(state)}")
    
    rag_results = state['RAG_results']
    web_results = state['web_search_results']
    question = state['question']
    
    logger.info(f"Question: {question}")
    logger.info(f"RAG results count: {len(rag_results)}")
    logger.info(f"Web results count: {len(web_results)}")
    
    logger.info("Creating structured LLM with Answer schema...")
    structured_llm = llm.with_structured_output(Answer)
    
    logger.info("Invoking LLM to generate final answer...")
    messages = [
        instructions,
        HumanMessage(content=f"Question: {question}\n\nRAG Results: {rag_results}\n\nWeb Results: {web_results}")
    ]
    answer = structured_llm.invoke(messages)
    logger.info(f"Generated answer: {truncate_log_content(answer.answer)}")
    
    result = {"answer": answer.answer}
    logger.info(f"EXITING generate_answer with result: {truncate_log_content(result)}")
    logger.info("="*50)
    
    return result



# Build search sub graph
logger.info("Building search sub-graph...")
search_builder = StateGraph(state_schema=SearchState, output_schema=SearchOut)
logger.info("Adding nodes to search sub-graph...")
search_builder.add_node("web_search", search_web)
search_builder.add_node("wikipedia_search", search_wikipedia)
logger.info("Adding edges to search sub-graph...")
search_builder.add_edge(START, "web_search")
search_builder.add_edge("web_search", "wikipedia_search")
search_builder.add_edge("wikipedia_search", END)
logger.info("Search sub-graph built successfully")

# Build RAG sub graph
logger.info("Building RAG sub-graph...")
rag_builder = StateGraph(state_schema=RAGState, output_schema=RAGOut)
logger.info("Adding nodes to RAG sub-graph...")
rag_builder.add_node("rag_retrieval", rag_retrieval)
rag_builder.add_node("rag_rerank", rag_rerank)
rag_builder.add_node("format_rag_results", format_rag_results)
rag_builder.add_node("should_call_agent", should_call_agent)
rag_builder.add_node("agent_entry", agent_entry)
rag_builder.add_node("rag_agent_check", rag_agent_check)
rag_builder.add_node("agent_gate", agent_gate, join=True)
rag_builder.add_node("rag_fuse_prune", rag_fuse_prune)

logger.info("Adding edges to RAG sub-graph...")
rag_builder.add_edge(START, "rag_retrieval")
rag_builder.add_edge("rag_retrieval", "rag_rerank")
rag_builder.add_edge("rag_rerank", "should_call_agent")
rag_builder.add_conditional_edges(
    "should_call_agent",
    route_after_should_call_agent,
)
rag_builder.add_conditional_edges("agent_entry", to_agent_check, ["rag_agent_check"])
rag_builder.add_edge("rag_agent_check", "agent_gate")
rag_builder.add_edge("agent_gate", "rag_fuse_prune")
rag_builder.add_edge("rag_fuse_prune", "format_rag_results")
rag_builder.add_edge("format_rag_results", END)
rag_graph = rag_builder.compile()
# Mermaid PNG
with open("rag_graph.png", "wb") as f:
    f.write(rag_graph.get_graph().draw_mermaid_png())
logger.info("RAG sub-graph built successfully")


# Build main graph
logger.info("Building main graph...")
main_builder = StateGraph(state_schema=OverallState)
logger.info("Adding nodes to main graph...")
main_builder.add_node("check_question", check_question)
logger.info("Compiling and adding search sub-graph as node...")
main_builder.add_node("search", search_builder.compile())
logger.info("Compiling and adding RAG sub-graph as node...")
main_builder.add_node("rag", rag_graph)
main_builder.add_node("generate_answer", generate_answer)

logger.info("Adding edges to main graph...")
main_builder.add_edge(START, "check_question")
main_builder.add_edge("check_question", "search")
main_builder.add_edge("check_question", "rag")
main_builder.add_edge("search", "generate_answer")
main_builder.add_edge("rag", "generate_answer")
main_builder.add_edge("generate_answer", END)

logger.info("Compiling main graph...")
# In-memory checkpointer (per-process). For persistence across restarts, switch to SqliteSaver.
checkpointer = MemorySaver()
graph = main_builder.compile(checkpointer=checkpointer)
logger.info("Main graph compiled successfully!")


#  write this to a file
logger.info("Generating graph visualization...")
with open("graph.png", "wb") as f:
    f.write(graph.get_graph(xray=1).draw_mermaid_png())
logger.info("Graph visualization saved to graph.png")


logger.info("\n" + "="*60)
logger.info("STARTING GRAPH EXECUTION")
logger.info("="*60)
question = "Describe the absorption pathway of vitamin B12 from mouth to ileum, including intrinsic factor, the site of uptake, and one clinical consequence of pernicious anemia."
logger.info(f"Input question: {question}")
logger.info("Invoking graph with question...")

# Unique thread for this run so checkpoints are isolated
thread_id = f"run-{uuid.uuid4().hex}"
logger.info(f"Thread ID for this run: {thread_id}")
final_state = graph.invoke(
    {"question": question, "f": 3},
    config={"configurable": {"thread_id": thread_id}}
)

logger.info("="*60)
logger.info("GRAPH EXECUTION COMPLETED")
logger.info(f"Final state: {truncate_log_content(final_state)}")
logger.info("="*60)

logger.info(f"\nFINAL ANSWER: {truncate_log_content(final_state['answer'])}")
print("\n" + "="*60)
print(f"ANSWER: {final_state['answer']}")
print("="*60)
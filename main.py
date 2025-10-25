from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
import operator
from typing import List, Optional, Annotated, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
# Web search tool
from langchain_tavily import TavilySearch
# Wikipedia search tool
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv
import os
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# Reranker
import re
# Optional Cohere reranker
import cohere
# Optional: use HF embeddings if you indexed with Sentence Transformers
# from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging to only print to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

# Question checking
check_instructions = SystemMessage(content=f"""You are a helpful assistant specialized in checking if a question is relevant to nutrition, fitness, health, and wellness.""")

class CheckQuestion(BaseModel):
    is_relevant: bool = Field(None, description="Whether the question is relevant to nutrition, fitness, health, and wellness.")

def check_question(state: OverallState):

    """ Check if the question is relevant to nutrition, fitness, health, and wellness """
    
    logger.info("="*50)
    logger.info("ENTERING check_question function")
    logger.info(f"State received: {state}")
    
    # Check question
    logger.info("Creating structured LLM with CheckQuestion schema...")
    structured_llm = llm.with_structured_output(CheckQuestion)
    
    logger.info("Invoking LLM to check question relevance...")
    messages = [check_instructions, HumanMessage(content=state['question'])]
    is_relevant = structured_llm.invoke(messages)
    
    logger.info(f"Question relevance result: {is_relevant.is_relevant}")
    result = {"is_relevant": is_relevant.is_relevant}
    logger.info(f"EXITING check_question with result: {result}")
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
    logger.info(f"State received: {state}")

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
    logger.info(f"Search result preview: {str(search_result)[:200]}...")
    
     # Format
    logger.info("Formatting web search results...")
    # TavilySearch returns a string directly, not a list of dicts
    if isinstance(search_result, str):
        formatted_search_docs = f'<Document source="Tavily Web Search"/>\n{search_result}\n</Document>'
    elif isinstance(search_result, list):
        # Handle if it returns a list
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc.get("url", "N/A")}"/>\n{doc.get("content", str(doc))}\n</Document>'
                for doc in search_result
            ]
        )
    else:
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
    logger.info(f"State received: {state}")
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
    logger.info(f"Found {len(search_docs)} Wikipedia documents")

     # Format
    logger.info("Formatting Wikipedia search results...")
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    logger.info(f"Formatted Wikipedia results (length: {len(formatted_search_docs)} chars)")
    
    result = {"web_search_results": [formatted_search_docs]}
    logger.info(f"EXITING search_wikipedia with {len(result['web_search_results'])} result(s)")
    logger.info("="*50)

    return result 

class RAGState(TypedDict):
    # Inputs (control)
    question: str
    k: int                 # number to retrieve
    m: int                 # number to keep after rerank
    # Internals (not required to exit the subgraph, but useful between nodes)
    candidates: List[Document]         
    reranked: List[Document]           
    RAG_results: Annotated[List[str], operator.add]
    # Optional: structured winners for UI, analytics, or debugging
    RAG_meta: Annotated[List[Dict], operator.add]

class RAGOut(TypedDict):
    RAG_results: Annotated[List[str], operator.add]
    RAG_meta:    Annotated[List[Dict], operator.add]

def rag_retrieval(state: RAGState):
    """Retrieve top-k candidate chunks from a persisted Chroma collection using the same embedding model as ingestion."""
    logger.info("=" * 50)
    logger.info("ENTERING rag_retrieval function")
    logger.info(f"State received: {state}")

    question = state["question"]
    k = state.get("k", 10)  # default recall size

    try:
        # 1) Build the embedding function that MATCHES your ingestion
        embeddings = OpenAIEmbeddings()  # match your ingestion model
        # 2) Open Chroma collection (created by your ingestion script)
        index_dir = "chroma-index"  # change if your index folder is elsewhere
        collection_name = "nutrition_knowledge_base"  # must match your ingestion collection
        logger.info(f"Opening Chroma collection from: {index_dir} (collection='{collection_name}')")
        vs = Chroma(
            embedding_function=embeddings,
            persist_directory=index_dir,
            collection_name=collection_name,
        )

        # 3) Create a retriever and fetch candidates (MMR + wider fetch_k)
        #    - k: how many you return to reranker
        #    - fetch_k: how many to consider before MMR trims redundancy
        fetch_k = max(50, k * 5)  # widen the net
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        logger.info(f"Retrieving (MMR) top-{k} from fetch_k={fetch_k} for question: {question!r}")
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} candidate docs")

        # 4) Return into state, for the rerank node to consume
        result = {"candidates": docs}
        logger.info(f"EXITING rag_retrieval with result count: {len(docs)}")
        logger.info("=" * 50)
        return result

    except Exception as e:
        logger.exception("rag_retrieval (Chroma) failed")
        # Fail-soft: return empty candidates so downstream can decide what to do
        return {"candidates": []}
    

def rag_rerank(state: RAGState):
    """Rerank candidates to top-m using Cohere SDK directly. No formatting."""
    logger.info("=" * 50)
    logger.info("ENTERING rag_rerank function")
    logger.info(f"State received (keys): {list(state.keys())}")

    question = state["question"]
    m = state.get("m", 5)
    candidates: List[Document] = state.get("candidates", []) or []

    if not candidates:
        logger.info("No candidates provided; returning empty reranked list.")
        return {"reranked": []}

    # Cohere limit is ~100 docs per call; we pass at most 'len(candidates)'
    top_n = min(m, len(candidates))

    try:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not set; falling back to retriever order.")
            return {"reranked": candidates[:m]}

        client = cohere.Client(api_key)
        docs_text = [d.page_content or "" for d in candidates]

        logger.info(f"Using Cohere ReRank (SDK) model=rerank-english-v3.0; docs={len(docs_text)}, top_n={top_n}")
        resp = client.rerank(
            model="rerank-english-v3.0",   # or "rerank-multilingual-v3.0"
            query=question,
            documents=docs_text,
            top_n=top_n,
        )

        # Map Cohere results (which include original indices) back to Documents
        ordered: List[Document] = []
        for r in resp.results:
            d = candidates[r.index]
            # Attach score for downstream inspection
            try:
                d.metadata["rerank_score"] = float(r.relevance_score)
            except Exception:
                d.metadata["rerank_score"] = None
            ordered.append(d)

        logger.info(f"Reranked {len(candidates)} → top-{len(ordered)} via Cohere SDK")
        return {"reranked": ordered}

    except Exception:
        logger.exception("rag_rerank (Cohere SDK) failed; returning fallback order.")
        return {"reranked": candidates[:m]}

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

        # Optional small log preview (also not truncated, but clipped in logs by slicing, not altering data)
        logger.info(f"RAG line preview: {line[:300]}{'…' if len(line) > 300 else ''}")

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
    logger.info(f"State received: {state}")
    
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
    logger.info(f"Generated answer: {answer.answer}")
    
    result = {"answer": answer.answer}
    logger.info(f"EXITING generate_answer with result: {result}")
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
logger.info("Adding edges to RAG sub-graph...")
rag_builder.add_edge(START, "rag_retrieval")
rag_builder.add_edge("rag_retrieval", "rag_rerank")
rag_builder.add_edge("rag_rerank", "format_rag_results")
rag_builder.add_edge("format_rag_results", END)
logger.info("RAG sub-graph built successfully")

# Build main graph
logger.info("Building main graph...")
main_builder = StateGraph(state_schema=OverallState)
logger.info("Adding nodes to main graph...")
main_builder.add_node("check_question", check_question)
logger.info("Compiling and adding search sub-graph as node...")
main_builder.add_node("search", search_builder.compile())
logger.info("Compiling and adding RAG sub-graph as node...")
main_builder.add_node("rag", rag_builder.compile())
main_builder.add_node("generate_answer", generate_answer)

logger.info("Adding edges to main graph...")
main_builder.add_edge(START, "check_question")
main_builder.add_edge("check_question", "search")
main_builder.add_edge("check_question", "rag")
main_builder.add_edge("search", "generate_answer")
main_builder.add_edge("rag", "generate_answer")
main_builder.add_edge("generate_answer", END)

logger.info("Compiling main graph...")
graph = main_builder.compile()
logger.info("Main graph compiled successfully!")


#  write this to a file
logger.info("Generating graph visualization...")
with open("graph.png", "wb") as f:
    f.write(graph.get_graph(xray=1).draw_mermaid_png())
logger.info("Graph visualization saved to graph.png")


logger.info("\n" + "="*60)
logger.info("STARTING GRAPH EXECUTION")
logger.info("="*60)
question = "Describe the enterohepatic circulation of bile acids and explain how it contributes to fat-soluble vitamin absorption."
logger.info(f"Input question: {question}")
logger.info("Invoking graph with question...")

final_state = graph.invoke({"question": question})

logger.info("="*60)
logger.info("GRAPH EXECUTION COMPLETED")
logger.info(f"Final state: {final_state}")
logger.info("="*60)

logger.info(f"\nFINAL ANSWER: {final_state['answer']}")
print("\n" + "="*60)
print(f"ANSWER: {final_state['answer']}")
print("="*60)
"""
Main LangGraph implementation for the Nutritionist Chatbot system.

This module defines the core chat graph that orchestrates the conversation flow:
1. Decision making: Determines which tools (RAG/Web) to use based on user input
2. Tool execution: Runs the selected tools in parallel
3. Answer synthesis: Combines tool results with conversation history to generate responses

The graph uses LangGraph's StateGraph with MessagesState for conversation management
and includes sophisticated routing logic to optimize tool usage.
"""

from __future__ import annotations
import logging
import os
import uuid
from typing import Annotated, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangGraph components for building the conversation graph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# LangChain components for LLM interactions and message handling
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the tool wrappers from our tools module
from tools import web_search_tool, rag_tool

# ----------------------------------------------------------------------------
# Logging and Environment Setup
# ----------------------------------------------------------------------------

# Configure logging for the main module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv()
logger.info("Environment variables loaded successfully")

# Initialize the LLM for classification and answer synthesis
# Using Gemini 2.0 Flash Experimental for fast, reliable responses
logger.info("Initializing ChatGoogleGenerativeAI model (gemini-2.0-flash-exp)...")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
logger.info("LLM model initialized successfully")

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def truncate_log_content(content, max_words: int = 12) -> str:
    """
    Truncate content for logging purposes to keep logs readable.
    
    Args:
        content: Content to truncate (any type)
        max_words: Maximum number of words to keep
        
    Returns:
        Truncated string representation of content
    """
    if content is None:
        return "<None>"
    try:
        s = str(content)
    except Exception:
        return "<unprintable>"
    words = s.split()
    return s if len(words) <= max_words else " ".join(words[:max_words]) + "..."

# ----------------------------------------------------------------------------
# State Management and Pydantic Schemas
# ----------------------------------------------------------------------------

class ChatState(MessagesState):
    """
    Global state for the main chat graph.
    
    Inherits `messages` from MessagesState for conversation history management.
    Additional fields track tool outputs and control parameters.
    
    Attributes:
        answer: The final generated answer (str | None)
        is_relevant: Whether the question is relevant to nutrition (bool | None)
        web_search_results: Results from web search tool (List[str])
        RAG_results: Results from RAG tool (List[str])
        f: Number of chunks to use for RAG (int, default 3)
        need_rag: Whether to use RAG tool (bool)
        need_web: Whether to use web search tool (bool)
    """
    # Outputs & metadata
    answer: str | None = None
    is_relevant: bool | None = None

    # Tool outputs
    web_search_results: List[str] = []
    RAG_results: List[str] = []

    # Single control knob: how many chunks flow into generate_answer
    f: int = 3

    # Tool routing flags
    need_rag: bool = False
    need_web: bool = False

def _last_user_text(messages):
    """
    Extract the text content from the most recent user message.
    
    Args:
        messages: List of message objects from the conversation
        
    Returns:
        Text content of the last user message, or empty string if none found
    """
    for m in reversed(messages or []):
        if getattr(m, "type", None) in ("human", "user") or getattr(m, "role", None) == "user":
            return getattr(m, "content", "") or ""
    return ""

class Decision(BaseModel):
    """
    Pydantic model for structured decision making about tool usage.
    
    Used by the LLM to determine which tools to call based on the user's question.
    """
    need_rag: bool = Field(..., description="Use the RAG tool?")
    need_web: bool = Field(..., description="Use the Web tool?")
    f: int = Field(3, ge=1, le=10, description="Final chunk budget for RAG")
    reason: str = Field("", description="Short rationale for the decision")

# System message for the decision-making LLM
# Defines the routing logic for determining which tools to use
decide_system = SystemMessage(content=(
    "You are routing a chat assistant. Decide whether to call domain RAG and/or Web.\n"
    "- If the user asks to rephrase, simplify, summarize, compare, or otherwise transform THEIR text, no tools are needed.\n"
    "- If the user asks a meta question about the conversation (e.g., 'what did I ask?'), no tools are needed.\n"
    "- For stable textbook/KB facts → prefer RAG.\n"
    "- For recent/timely facts or things unlikely in the KB → use Web.\n"
    "- Keep f small unless the question is multi-part or ambiguous.\n"
    "Return a brief reason."
))

def decide(state: ChatState):
    """
    Decision-making node that determines which tools to use based on user input.
    
    This is the first node in the graph that analyzes the user's question and decides:
    - Whether to use RAG (for domain knowledge)
    - Whether to use Web search (for recent/timely information)
    - How many chunks to retrieve (f parameter)
    
    Args:
        state: Current chat state containing messages and other data
        
    Returns:
        Dictionary with tool routing decisions and parameters
    """
    logger.info("=" * 50)
    logger.info("ENTERING decide function")
    
    # Extract the user's question from the most recent message
    user_q = _last_user_text(state.get("messages", []))
    
    # Use structured output to get consistent decision format
    structured = llm.with_structured_output(Decision)
    res = structured.invoke([decide_system, HumanMessage(content=user_q)])
    
    # Validate and constrain the f parameter
    f = max(1, min(int(res.f or 3), 10))
    
    # Prepare output with routing decisions
    out = {
        "is_relevant": True,  # Assume all questions are relevant for now
        "need_rag": bool(res.need_rag),
        "need_web": bool(res.need_web),
        "f": f,
    }
    
    logger.info(f"DECIDE → need_rag={out['need_rag']} need_web={out['need_web']} f={out['f']} reason={res.reason}")
    logger.info("=" * 50)
    return out

# ----------------------------------------------------------------------------
# Tool Bridge Nodes (Thin Wrappers)
# ----------------------------------------------------------------------------
# These nodes act as bridges between the LangGraph state and our tool functions

def run_web_tool_node(state: ChatState):
    """
    Bridge node for web search tool execution.
    
    Checks if web search is needed and calls the web_search_tool if so.
    Handles errors gracefully by returning empty results.
    
    Args:
        state: Current chat state with routing decisions
        
    Returns:
        Dictionary with web search results or empty list
    """
    # Skip if web search was not requested
    if not state.get("need_web", False):
        logger.info("web_tool: skipped (need_web=False)")
        return {"web_search_results": []}
        
    # Extract user question and call web search tool
    q: str = _last_user_text(state.get("messages", []))
    logger.info("run_web_tool_node: calling web_search_tool")
    
    try:
        res = web_search_tool(q)
        return {"web_search_results": res.get("web_search_results", [])}
    except Exception:
        logger.exception("web_search_tool failed; returning empty list")
        return {"web_search_results": []}


def run_rag_tool_node(state: ChatState):
    """
    Bridge node for RAG tool execution.
    
    Checks if RAG is needed and calls the rag_tool with the specified chunk budget.
    Handles errors gracefully by returning empty results.
    
    Args:
        state: Current chat state with routing decisions and f parameter
        
    Returns:
        Dictionary with RAG results or empty list
    """
    # Skip if RAG was not requested
    if not state.get("need_rag", False):
        logger.info("rag_tool: skipped (need_rag=False)")
        return {"RAG_results": []}
        
    # Extract user question and chunk budget
    q: str = _last_user_text(state.get("messages", []))
    f: int = int(state.get("f", 3) or 3)
    logger.info("run_rag_tool_node: calling rag_tool")
    
    try:
        res = rag_tool(q, f=f)
        return {"RAG_results": res.get("RAG_results", [])}
    except Exception:
        logger.exception("rag_tool failed; returning empty outputs")
        return {"RAG_results": []}

# ----------------------------------------------------------------------------
# Final Answer Generation
# ----------------------------------------------------------------------------

# System instructions for the answer generation LLM
# Defines two modes: tool-backed factual mode and history-only mode
instructions = SystemMessage(
    content=(
        "You are a helpful chat assistant. You can:\n"
        "• Answer questions using provided tool results (RAG/Web) and conversation history.\n"
        "• Handle meta questions about this chat (e.g., 'what did I ask?').\n"
        "• Transform user text (rephrase, simplify, summarize) without adding new facts.\n\n"

        "MODES\n"
        "1) TOOL-BACKED FACTUAL MODE (use when RAG/Web Results are provided):\n"
        "   - Make factual claims grounded in those results and conversation history.\n"
        "   - PRIORITIZE RAG over Web if they conflict.\n"
        "   - Add inline citations immediately after claims sourced from RAG/Web.\n"
        "   - End with a 'References' section listing each unique source you used.\n\n"
        "2) HISTORY-ONLY MODE (no tool results provided):\n"
        "   - You may use the conversation history to answer meta questions or transform user text.\n"
        "   - Do NOT invent new factual claims beyond what the user already provided in this chat.\n"
        "   - If a factual answer requires information not in the conversation, reply exactly: 'I don't know'.\n"
        "   - No citations are required in this mode.\n"
    )
)

class Answer(BaseModel):
    """
    Pydantic model for structured answer generation.
    
    Ensures the LLM returns a properly formatted answer string.
    """
    answer: str = Field(None, description="Answer to the question.")


def generate_answer(state: ChatState):
    """
    Final answer generation node that synthesizes tool results with conversation history.
    
    This is the final node in the graph that:
    1. Extracts tool results and conversation history
    2. Determines the appropriate mode (tool-backed vs history-only)
    3. Generates a comprehensive answer using the LLM
    4. Returns the answer and adds it to the conversation
    
    Args:
        state: Current chat state with tool results and conversation history
        
    Returns:
        Dictionary with the generated answer and updated messages
    """
    logger.info("=" * 50)
    logger.info("ENTERING generate_answer function")
    logger.info(f"State received: {truncate_log_content(state)}")

    # Extract data from state
    rag_results = state.get("RAG_results", [])
    web_results = state.get("web_search_results", [])
    user_text = _last_user_text(state.get("messages", []))
    conversation_history = state.get("messages", [])

    logger.info(f"User message: {user_text}")
    logger.info(f"Conversation history length: {len(conversation_history)} messages")
    logger.info(f"RAG results count: {len(rag_results)}")
    logger.info(f"Web results count: {len(web_results)}")

    # Initialize structured output generator
    structured = llm.with_structured_output(Answer)

    # Build the prompt for answer generation
    messages = [instructions]

    # Include a rolling window of conversation history to provide context
    # This prevents prompt bloat while maintaining relevant context
    if len(conversation_history) > 1:
        window = conversation_history[-10:-1]  # Last 9 prior messages (exclude current user message)
        messages.extend(window)

    # Determine the mode based on available tool results
    tool_mode = "TOOL" if (rag_results or web_results) else "HISTORY_ONLY"

    # Create the final user message with context and mode information
    messages.append(HumanMessage(
        content=(
            f"MODE: {tool_mode}\n\n"
            f"Question: {user_text}\n\n"
            f"RAG Results: {rag_results}\n\n"
            f"Web Results: {web_results}"
        )
    ))

    # Generate the answer using the LLM
    ans = structured.invoke(messages)
    logger.info(f"Generated answer: {truncate_log_content(ans.answer)}")

    # Prepare output with answer and updated conversation
    out = {
        "answer": ans.answer,
        "messages": [AIMessage(content=ans.answer)],
    }
    logger.info(f"EXITING generate_answer with result: {truncate_log_content(out)}")
    logger.info("=" * 50)
    return out

# ----------------------------------------------------------------------------
# Graph Construction and Compilation
# ----------------------------------------------------------------------------

logger.info("Building main graph (clean)")

# Create the StateGraph with our ChatState schema
main_builder = StateGraph(state_schema=ChatState)

# Add nodes to the graph
# Each node represents a step in the conversation flow
main_builder.add_node("decide", decide)           # Decision making
main_builder.add_node("web", run_web_tool_node)   # Web search execution
main_builder.add_node("rag", run_rag_tool_node)   # RAG execution
main_builder.add_node("synthesize", generate_answer, join=True)  # Answer synthesis (joins all inputs)

# Define the graph flow
# START -> decide -> (web, rag) -> synthesize -> END
main_builder.add_edge(START, "decide")           # Start with decision making
main_builder.add_edge("decide", "web")           # Decision leads to web tool
main_builder.add_edge("decide", "rag")           # Decision leads to RAG tool
main_builder.add_edge("web", "synthesize")       # Web results go to synthesis
main_builder.add_edge("rag", "synthesize")       # RAG results go to synthesis
main_builder.add_edge("synthesize", END)         # Synthesis completes the flow

# Compile the graph with an in-memory checkpointer for conversation persistence
checkpointer = MemorySaver()
graph = main_builder.compile(checkpointer=checkpointer)
logger.info("Main graph compiled successfully!")

# ----------------------------------------------------------------------------
# Main Execution Block (for testing and demonstration)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate a visual representation of the graph for documentation
    logger.info("Generating graph visualization...")
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph(xray=1).draw_mermaid_png())
    logger.info("Graph visualization saved to graph.png")

    # Example execution with a nutrition question
    logger.info("\n" + "=" * 60)
    logger.info("STARTING GRAPH EXECUTION")
    logger.info("=" * 60)

    # Define a complex nutrition question for testing
    question = (
        "Describe the absorption pathway of vitamin B12 from mouth to ileum, "
        "including intrinsic factor, the site of uptake, and one clinical consequence of pernicious anemia."
    )
    logger.info(f"User message: {question}")

    # Generate a unique thread ID for this conversation
    thread_id = f"run-{uuid.uuid4().hex}"
    logger.info(f"Thread ID for this run: {thread_id}")

    # Prepare the input using MessagesState format
    # This maintains conversation history while providing the current question
    messages = [HumanMessage(content=question)]

    # Execute the graph with the question
    # The graph will:
    # 1. Decide which tools to use (likely RAG for this domain question)
    # 2. Execute the selected tools in parallel
    # 3. Synthesize the results into a comprehensive answer
    final_state = graph.invoke(
        {"messages": messages, "f": 3},  # Start with 3 chunks for RAG
        config={"configurable": {"thread_id": thread_id}},
    )

    # Log the execution results
    logger.info("=" * 60)
    logger.info("GRAPH EXECUTION COMPLETED")
    logger.info(f"Final state: {truncate_log_content(final_state)}")
    logger.info("=" * 60)

    # Display the final answer
    logger.info(f"\nFINAL ANSWER: {truncate_log_content(final_state['answer'])}")
    print("\n" + "=" * 60)
    print(f"ANSWER: {final_state['answer']}")
    print("=" * 60)
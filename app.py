#!/usr/bin/env python3
"""
Flask web server for the Nutritionist Chatbot system.

This module provides a simple web interface to interact with the LangGraph-based
chatbot system. It exposes REST API endpoints for chat functionality and serves
a web-based chat interface.

The Flask app acts as a bridge between web clients and the sophisticated
LangGraph conversation system defined in main.py.

Key Features:
- REST API for chat interactions
- Session management with thread IDs
- Error handling and logging
- Web interface for easy testing
"""

from flask import Flask, render_template, request, jsonify
import uuid
import logging

# Configure logging for the Flask application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Import the compiled graph from main.py
# Note: This will execute main.py on first import, building the graph
logger.info("Importing graph from main.py...")
from main import graph
from langchain_core.messages import HumanMessage
logger.info("Graph imported successfully!")

# ----------------------------------------------------------------------------
# Flask Route Handlers
# ----------------------------------------------------------------------------

@app.route('/')
def home():
    """
    Serve the main chat interface.
    
    Returns:
        Rendered HTML template for the chat interface
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests via REST API.
    
    This endpoint processes user questions through the LangGraph system and returns
    AI-generated responses. It supports session management through thread IDs
    to maintain conversation context.
    
    Expected JSON Input:
        {
            "question": "user's question",
            "thread_id": "optional session identifier"
        }
    
    Returns JSON Response:
        Success: {"answer": "AI response", "error": null}
        Error: {"answer": null, "error": "error message"}
    
    HTTP Status Codes:
        200: Success
        400: Bad request (missing question)
        500: Internal server error
    """
    try:
        # Extract and validate input data
        data = request.get_json()
        question = data.get('question', '').strip()
        thread_id = data.get('thread_id', '').strip()
        
        # Validate required fields
        if not question:
            return jsonify({
                'answer': None,
                'error': 'Please provide a question'
            }), 400
        
        # Use provided thread_id or generate a new one for session management
        if not thread_id:
            thread_id = f"web-{uuid.uuid4().hex}"
        
        logger.info(f"Received question: {question} (thread: {thread_id})")
        
        # Execute the LangGraph conversation system
        # The graph will:
        # 1. Analyze the question and decide which tools to use
        # 2. Execute RAG and/or web search tools as needed
        # 3. Synthesize results into a comprehensive answer
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)], "f": 3},
            config={
                "configurable": {"thread_id": thread_id},
                "max_concurrency": 3  # Limit parallel LLM calls to avoid rate limits
            }
        )
        
        # Extract the generated answer
        answer = result.get('answer', 'I apologize, but I could not generate an answer.')
        logger.info(f"Generated answer (length: {len(answer)} chars)")
        
        # Return successful response
        return jsonify({
            'answer': answer,
            'error': None
        })
    
    except Exception as e:
        # Log the error and return a user-friendly error message
        logger.exception("Error processing chat request")
        return jsonify({
            'answer': None,
            'error': f'An error occurred: {str(e)}'
        }), 500

# ----------------------------------------------------------------------------
# Application Entry Point
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # Display startup banner with application information
    print("\n" + "="*60)
    print("🍎 Nutritionist Chatbot Server")
    print("="*60)
    print("Starting Flask server on http://127.0.0.1:5000")
    print("Open this URL in your browser to use the chatbot")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Start the Flask development server
    # Note: debug=True enables auto-reload and detailed error pages
    # In production, set debug=False and use a proper WSGI server
    app.run(debug=True, host='127.0.0.1', port=5000)


#!/usr/bin/env python3
"""
Flask web server for the Nutritionist Chatbot
Simple interface to interact with the LangGraph-based chatbot system
"""

from flask import Flask, render_template, request, jsonify
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Import the compiled graph from main.py
# Note: This will execute main.py on first import, building the graph
logger.info("Importing graph from main.py...")
from main import graph
logger.info("Graph imported successfully!")

@app.route('/')
def home():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests
    Expects JSON: {"question": "user's question"}
    Returns JSON: {"answer": "bot's response", "error": null} or {"answer": null, "error": "error message"}
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        thread_id = data.get('thread_id', '').strip()
        
        if not question:
            return jsonify({
                'answer': None,
                'error': 'Please provide a question'
            }), 400
        
        # Use provided thread_id or generate a new one
        if not thread_id:
            thread_id = f"web-{uuid.uuid4().hex}"
        
        logger.info(f"Received question: {question} (thread: {thread_id})")
        
        # Invoke the graph with the question
        # Set max_concurrency to 3 to avoid hitting rate limits on parallel LLM calls
        result = graph.invoke(
            {"question": question, "f": 3},
            config={
                "configurable": {"thread_id": thread_id},
                "max_concurrency": 3
            }
        )
        
        answer = result.get('answer', 'I apologize, but I could not generate an answer.')
        logger.info(f"Generated answer (length: {len(answer)} chars)")
        
        return jsonify({
            'answer': answer,
            'error': None
        })
    
    except Exception as e:
        logger.exception("Error processing chat request")
        return jsonify({
            'answer': None,
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🍎 Nutritionist Chatbot Server")
    print("="*60)
    print("Starting Flask server on http://127.0.0.1:5000")
    print("Open this URL in your browser to use the chatbot")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)


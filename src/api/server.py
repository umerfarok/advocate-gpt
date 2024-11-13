# server.py
from flask import Flask, request, jsonify
from model.embeddings_manager import EmbeddingsManager
from model.qa_model import QASystem
from utils.memory_utils import MemoryManager
import time

app = Flask(__name__)

# Initialize systems
print("Initializing systems...")
embeddings_manager = EmbeddingsManager()
qa_system = QASystem()

# Load embeddings
embeddings_manager.load('vector_store')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy',
                   'device': qa_system.device,
                   'memory_available': MemoryManager.get_available_memory()})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        start_time = time.time()
        data = request.get_json()
        question = data['question']
        
        # Get relevant contexts
        relevant_docs = embeddings_manager.search(question, k=3)
        
        # Combine contexts
        combined_context = "\n".join([doc['text'] for doc in relevant_docs])
        
        # Generate answer
        answer = qa_system.generate_answer(question, combined_context)
        
        # Prepare response
        response = {
            'answer': answer,
            'sources': [{'source': doc['source'], 'relevance': doc['score']} 
                       for doc in relevant_docs],
            'processing_time': f"{time.time() - start_time:.2f} seconds"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/memory', methods=['GET'])
def memory_status():
    return jsonify({
        'ram_available': f"{MemoryManager.get_available_memory():.2f}GB",
        'gpu_available': f"{MemoryManager.get_gpu_memory():.2f}GB" 
                        if torch.cuda.is_available() else "N/A"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
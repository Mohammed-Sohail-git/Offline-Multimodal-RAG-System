"""
app.py - FIXED Flask UI for Multimodal RAG System with Accurate Results
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import uuid
from dataclasses import dataclass

# Import your pipeline components
from multimodal_unified_parallel import ParallelMasterPipeline
from ragchatbot import CrossModalRAGChatbot

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'change-this-secret-key-in-production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed_output'
app.config['CLIP_MODEL_PATH'] = 'SIH/clip-model'

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'doc',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp',
    'mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'wma', 'aac'
}

# Global state
chatbot_instance = None
processing_lock = False
current_session_id = None

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

if not os.path.exists("SIH"):
    print("❌ SIH directory not found. Run 'python download_models.py' first.")
    exit(1)

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    whisper_model: str = "SIH/faster-whisper-small" if os.path.exists("SIH/faster-whisper-small") else None

    def __post_init__(self):
        if self.whisper_model is None:
            raise RuntimeError("Audio model not found. Run 'python download_models.py'")

def load_html():
    """Load the HTML file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
        <h1>Error: index.html not found</h1>
        <p>Please ensure index.html is in the same directory as app.py</p>
        </body>
        </html>
        """


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_all_data():
    """Clear all uploaded and processed files."""
    global chatbot_instance, current_session_id
    
    # Clear uploads
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Clear processed
    if os.path.exists(app.config['PROCESSED_FOLDER']):
        shutil.rmtree(app.config['PROCESSED_FOLDER'])
        os.makedirs(app.config['PROCESSED_FOLDER'])
    
    chatbot_instance = None
    current_session_id = None


@app.route('/')
def index():
    """Serve the main HTML page."""
    html_content = load_html()
    return render_template_string(html_content)


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and processing."""
    global chatbot_instance, processing_lock, current_session_id
    
    try:
        print("\n" + "="*70)
        print("📤 UPLOAD REQUEST RECEIVED")
        print("="*70)
        
        # Check if already processing
        if processing_lock:
            print("⚠️  System busy")
            return jsonify({
                'success': False,
                'error': 'System is busy processing files. Please wait.'
            }), 429
        
        # Check files
        if 'files' not in request.files:
            print("❌ No files in request")
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        print(f"📦 Received {len(files)} files")
        
        if not files or files[0].filename == '':
            print("❌ No files selected")
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Start processing
        processing_lock = True
        
        # Clear previous data
        print("🧹 Clearing previous data...")
        clear_all_data()
        
        # Create new session
        current_session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_session_id)
        os.makedirs(session_dir, exist_ok=True)
        print(f"📁 Session: {current_session_id}")
        
        # Save uploaded files
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                print(f"   ✅ Saved: {filename}")
        
        if not uploaded_files:
            processing_lock = False
            print("❌ No valid files")
            return jsonify({
                'success': False,
                'error': 'No valid files uploaded'
            }), 400
        
        print(f"\n🚀 Processing {len(uploaded_files)} files...")
        
        # Process files
        pipeline = ParallelMasterPipeline(
            clip_model_path=app.config['CLIP_MODEL_PATH'],
            output_dir=app.config['PROCESSED_FOLDER'],
            max_workers=None
        )
        
        results = pipeline.run_complete_pipeline(session_dir)
        
        # Initialize chatbot with FULL capabilities
        print("\n🤖 Initializing enhanced chatbot...")
        chatbot_instance = CrossModalRAGChatbot(
            index_path=results['index_path']
        )
        
        processing_lock = False
        
        print("\n✅ UPLOAD & PROCESSING COMPLETE")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': f'Processed {results["total_chunks"]} chunks successfully',
            'files': uploaded_files,
            'total_chunks': results['total_chunks'],
            'duration': f"{results['duration_seconds']:.1f}s"
        }), 200
    
    except Exception as e:
        processing_lock = False
        print(f"\n❌ UPLOAD FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/query', methods=['POST'])
def query():
    """
    Handle text queries with ENHANCED accuracy.
    
    KEY IMPROVEMENTS:
    1. Uses hybrid search with reranking (k=15 for better coverage)
    2. Leverages full context formatting from ragchatbot.py
    3. Proper error handling and logging
    4. Uses the comprehensive prompt that works in CLI
    5. Returns debug info for transparency
    """
    global chatbot_instance
    
    try:
        print("\n" + "="*70)
        print("💬 QUERY REQUEST (ENHANCED)")
        print("="*70)
        
        if chatbot_instance is None:
            print("❌ Chatbot not initialized")
            return jsonify({
                'success': False,
                'error': 'Please upload and process files first'
            }), 400
        
        # Get JSON data
        data = request.get_json(silent=True)
        if not data:
            print("❌ No JSON data")
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        query_text = data.get('query', '').strip()
        debug_mode = data.get('debug', False)  # Optional debug flag
        
        if not query_text:
            print("❌ Empty query")
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        print(f"🔍 Query: {query_text}")
        print(f"🐛 Debug mode: {debug_mode}")
        
        # CRITICAL FIX: Use the SAME method that works in CLI
        # This uses:
        # - Hybrid search (Semantic + BM25 + Cross-encoder reranking)
        # - Proper context formatting by modality
        # - The comprehensive prompt that produces good results
        print("🔍 Performing hybrid search with reranking...")
        
        # Use k=7 for better performance instead of 15
        answer = chatbot_instance.chat_text(query_text, k=7)
        
        # Get search results for debug info
        if debug_mode:
            search_results = chatbot_instance.search_by_text(query_text, k=7)
            context = chatbot_instance.format_context_by_modality(search_results)
            
            debug_info = {
                'num_results': len(search_results),
                'context_breakdown': {
                    'text_chunks': len(context['text']),
                    'tables': len(context['tables']),
                    'images': len(context['images']),
                    'image_files': len(context['image_paths']),
                    'audio': len(context['audio'])
                },
                'top_sources': list(set([
                    r['metadata'].get('source', 'Unknown') 
                    for r in search_results[:5]
                ])),
                'top_scores': [
                    f"{r['score']:.4f}" for r in search_results[:5]
                ]
            }
            print(f"🐛 Debug info: {debug_info}")
        
        print(f"✅ Answer generated ({len(answer)} chars)")
        print("="*70 + "\n")
        
        response = {
            'success': True,
            'answer': answer
        }
        
        if debug_mode:
            response['debug'] = debug_info
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"\n❌ QUERY FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error message
        error_msg = str(e)
        if "ollama" in error_msg.lower():
            error_msg = "❌ Ollama is not running. Please start it with: ollama serve"
        elif "connection" in error_msg.lower():
            error_msg = "❌ Cannot connect to Ollama. Ensure it's running on localhost:11434"
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/clear', methods=['POST'])
def clear():
    """Clear all data."""
    global chatbot_instance, processing_lock
    
    try:
        print("\n" + "="*70)
        print("🧹 CLEAR REQUEST")
        print("="*70)
        
        if processing_lock:
            print("⚠️  Cannot clear while processing")
            return jsonify({
                'success': False,
                'error': 'Cannot clear while processing'
            }), 429
        
        clear_all_data()
        
        print("✅ All data cleared")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': 'All data cleared successfully'
        }), 200
    
    except Exception as e:
        print(f"\n❌ CLEAR FAILED: {e}\n")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Get system status with detailed info."""
    try:
        status_info = {
            'ready': chatbot_instance is not None,
            'processing': processing_lock,
            'session_id': current_session_id
        }
        
        # Add index statistics if chatbot is ready
        if chatbot_instance is not None:
            from collections import Counter
            metadata = chatbot_instance.metadata
            
            types = Counter(m['type'] for m in metadata)
            modalities = Counter(m.get('modality', 'text') for m in metadata)
            
            status_info['statistics'] = {
                'total_chunks': len(metadata),
                'by_type': dict(types),
                'by_modality': dict(modalities)
            }
        
        return jsonify(status_info), 200
    
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500


# Error handlers that return JSON
@app.errorhandler(404)
def not_found(e):
    print(f"❌ 404 Error: {request.path}")
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def internal_error(e):
    print(f"❌ 500 Error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large (max 500MB)'
    }), 413


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ENHANCED MULTIMODAL RAG CHATBOT SERVER")
    print("="*70)
    print("\n✨ KEY IMPROVEMENTS:")
    print("  ✅ Uses same search method as CLI (hybrid + reranking)")
    print("  ✅ Comprehensive prompt engineering")
    print("  ✅ Better context formatting by modality")
    print("  ✅ Enhanced error handling")
    print("  ✅ Detailed logging for debugging")
    print("\nFeatures:")
    print("  ✅ PDF, DOCX, Images, Audio support")
    print("  ✅ Parallel processing")
    print("  ✅ Cross-modal retrieval")
    print("  ✅ Natural language answers with citations")
    print("\nEndpoints:")
    print("  GET  /          - Main UI")
    print("  POST /upload    - Upload & process files")
    print("  POST /query     - Ask questions (ENHANCED)")
    print("  POST /clear     - Clear all data")
    print("  GET  /status    - System status")
    print("\n🌐 Access at: http://localhost:5000")
    print("="*70 + "\n")
    
    # Check if index.html exists
    if not os.path.exists('index.html'):
        print("⚠️  WARNING: index.html not found in current directory!")
        print("   Please ensure index.html is in the same folder as app.py\n")
    
    # Check if Ollama is running
    try:
        import ollama
        ollama.list()
        print("✅ Ollama is running")
    except:
        print("⚠️  WARNING: Ollama might not be running!")
        print("   Start it with: ollama serve")
        print("   Pull LLaVA with: ollama pull llava:7b\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
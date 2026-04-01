import os
import random
import string
import json
import traceback
from flask import Flask, request, jsonify, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from dotenv import load_dotenv

from database import init_db, get_db
import gridfs
from bson.objectid import ObjectId
import datetime
from email_utils_brevo import send_real_otp_email
from rag_engine import (
    extract_and_chunk_pdf, build_vector_store, retrieve_top_chunks,
    generate_plain_llm_answer, generate_rag_answer, calculate_faithfulness,
    calculate_general_accuracy, generate_quiz, generate_scholar_info, fetch_live_opportunities
)

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 100 MB."}), 413



# Initialize Database on startup
init_db()

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "ScholarRAG (ChatGPT Style)"})

# ================================
# Frontend Serving Routes
# ================================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat.html', methods=['GET'])
def chat_page():
    return render_template('chat.html')

@app.route('/dashboard.html', methods=['GET'])
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/home.html', methods=['GET'])
def home_page():
    return render_template('home.html')

# @app.route('/quiz.html', methods=['GET'])
# def quiz_page():
#     return render_template('quiz.html')

@app.route('/scholarships.html', methods=['GET'])
def scholarships_page():
    return render_template('scholarships.html')

@app.route('/compare.html', methods=['GET'])
def compare_page():
    return render_template('compare.html')

@app.route('/learn.html', methods=['GET'])
def learn_page():
    return render_template('learn.html')

# @app.route('/quiz/generate', methods=['POST'])
# def quiz_generate():
#     data = request.json
#     topic = data.get('topic')
#     level = data.get('level', 'Medium')
#     qty = int(data.get('qty', 3))
#     
#     if not topic:
#         return jsonify({"error": "Topic is required"}), 400
#         
#     try:
#         json_str = generate_quiz(topic, level, qty)
#         questions = json.loads(json_str)
#         return jsonify({"questions": questions})
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": "Failed to parse AI output. AI may have returned invalid JSON."}), 500


# ================================
# Auth Endpoints (SMTP Updated)
# ================================
@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    otp = generate_otp()
    password_hash = generate_password_hash(password)
    
    db = get_db()
    users_coll = db.users
    try:
        # Check if user exists
        existing_user = users_coll.find_one({'email': email})
        if existing_user:
            users_coll.update_one({'email': email}, {'$set': {'password_hash': password_hash, 'otp': otp, 'verified': False}})
        else:
            users_coll.insert_one({'email': email, 'password_hash': password_hash, 'otp': otp, 'verified': False})
    except Exception as e:
        pass
        
    email_sent = send_real_otp_email(email, otp)
    if email_sent:
        return jsonify({"message": "OTP sent successfully! Please check your email inbox and spam folder."})
    else:
        return jsonify({"error": "Failed to dispatch email. Please check server configuration."}), 500

@app.route('/auth/verify', methods=['POST'])
def verify():
    data = request.json
    email = data.get('email')
    otp = data.get('otp')
    
    db = get_db()
    users_coll = db.users
    user = users_coll.find_one({'email': email, 'otp': otp})
    
    if user:
        users_coll.update_one({'email': email}, {'$set': {'verified': True, 'otp': None}})
        return jsonify({"message": "Verification successful!", "user_id": str(user['_id'])})
    
    return jsonify({"error": "Invalid OTP or Email"}), 401

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
        
    db = get_db()
    users_coll = db.users
    user = users_coll.find_one({'email': email})
    
    if user and user['verified']:
        if user.get('password_hash') and check_password_hash(user['password_hash'], password):
            return jsonify({"message": "Login successful.", "user_id": str(user['_id'])})
        else:
            return jsonify({"error": "Invalid email or password"}), 401
    elif user and not user['verified']:
        return jsonify({"error": "Account not verified. Please complete signup."}), 403
    else:
        return jsonify({"error": "User not found"}), 404


# ================================
# Chat History Management Endpoints
# ================================
@app.route('/chat/sessions/<user_id>', methods=['GET'])
def get_sessions(user_id):
    db = get_db()
    sessions = list(db.chat_sessions.find({'user_id': str(user_id)}).sort('created_at', -1))
    for s in sessions:
        s['id'] = str(s['_id'])
        del s['_id']
    return jsonify(sessions)

@app.route('/chat/session/new', methods=['POST'])
def create_session():
    data = request.json
    user_id = data.get('user_id')
    title = data.get('title', 'New Chat')
    
    if not user_id:
        return jsonify({"error": "User ID required"}), 400
        
    db = get_db()
    result = db.chat_sessions.insert_one({
        'user_id': str(user_id),
        'title': title,
        'created_at': datetime.datetime.utcnow()
    })
    session_id = str(result.inserted_id)
    return jsonify({"session_id": session_id, "title": title})

@app.route('/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    db = get_db()
    messages = list(db.chat_history.find({'session_id': session_id}).sort('timestamp', 1))
    for m in messages:
        m['id'] = str(m['_id'])
        del m['_id']
    return jsonify(messages)

# ================================
# Primary ChatGPT RAG Inference
# ================================
@app.route('/chat/message', methods=['POST'])
def chat_message():
    data = request.json
    session_id = data.get('session_id')
    user_id = data.get('user_id') # For safety
    query = data.get('query')

    if not session_id or not query:
        return jsonify({"error": "Session ID and Query required"}), 400

    db = get_db()
    
    # 1. Store User Query
    db.chat_history.insert_one({
        'session_id': session_id,
        'role': 'user',
        'content': query,
        'timestamp': datetime.datetime.utcnow()
    })
    
    # 2. Retrieve Past Chat History (limit to last 10 messages for context brevity)
    history_rows = list(db.chat_history.find({'session_id': session_id}).sort('timestamp', 1).limit(10))
    
    formatted_history = ""
    for row in history_rows[:-1]:  # Exclude the current query we just added
        formatted_history += f"{'Student' if row['role'] == 'user' else 'Tutor'}: {row['content']}\\n\\n"
    
    # Update title dynamically on the first message
    if len(history_rows) == 1:
        new_title = query[:30] + "..." if len(query) > 30 else query
        try:
            db.chat_sessions.update_one({'_id': ObjectId(session_id)}, {'$set': {'title': new_title}})
        except:
            pass

    # 3. Retrieve Context from ChromaDB (with timeout to avoid hanging on embedding API)
    import concurrent.futures
    chunks = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(retrieve_top_chunks, query)
            chunks = future.result(timeout=30)  # 30 second timeout on embedding lookup
    except concurrent.futures.TimeoutError:
        print("[chat_message] Embedding retrieval timed out — proceeding without RAG context.")
    except Exception as retrieval_err:
        print(f"[chat_message] Retrieval error (non-fatal): {retrieval_err}")

    context_text = ""
    if chunks:
        for doc in chunks:
            unit = doc.metadata.get('unit_number', 'Unknown')
            subject = doc.metadata.get('subject_name', 'Unknown')
            context_text += f"[Subject: {subject} | Unit: {unit}] {doc.page_content}\\n\\n"
    
    # 4. Generate AI Response (Injected with History and RAG Context)
    system_prompt = f"""You are a helpful, scholarly AI Tutor. Answer the latest question using previous chat history and syllabus notes.
    
    CRITICAL INSTRUCTIONS FOR YOUR ANSWER:
    1. You MUST provide your answer in concise bullet points.
    2. Do NOT write long paragraphs. Keep the explanation brief to save generation time.
    3. Only include directly relevant information to the question asked.

    Previous Conversation:
    {formatted_history}

    Syllabus Notes:
    {context_text}
    """

    user_prompt = f"Question: {query}\\nAnswer:"

    # Call Gemini directly with explicit timeout and Round-Robin rotation
    import rag_engine
    max_attempts = max(1, len(getattr(rag_engine, 'API_KEYS', []))) * len(getattr(rag_engine, 'FALLBACK_MODELS', [1]))
    attempts = 0
    ai_response = "Sorry, all AI models are currently overloaded. Please try again later."

    while attempts < max_attempts:
        try:
            model = rag_engine.get_gemini_model()
            if model is None:
                raise Exception("Gemini model not initialized. Check GEMINI_API_KEY in environment.")
            res = model.generate_content(
                system_prompt + "\\n" + user_prompt,
                request_options={"timeout": 60}
            )
            ai_response = res.text
            break # Success, we got a response!
        except Exception as e:
            error_msg = str(e).lower()
            if "404" in error_msg or "429" in error_msg or "503" in error_msg or "504" in error_msg or "quota" in error_msg or "resourceexhausted" in error_msg or "deadline" in error_msg:
                print(f"[chat_message] API Quota/Overload on attempt {attempts+1}. Rotating key/model...")
                rag_engine.rotate_api_key()
                attempts += 1
                if attempts == max_attempts:
                    print("[chat_message] Exhausted all keys and models.")
                    ai_response = "Sorry, I encountered an error answering that. Detail: " + error_msg
            else:
                traceback.print_exc()
                ai_response = "Sorry, I encountered an error answering that. Detail: " + error_msg
                break

    # 5. Store AI Response
    db.chat_history.insert_one({
        'session_id': session_id,
        'role': 'ai',
        'content': ai_response,
        'timestamp': datetime.datetime.utcnow()
    })

    return jsonify({
        "role": "ai",
        "content": ai_response,
        "context_sources": len(chunks)
    })

# ================================
# 4. RAG Endpoints
# ================================

@app.route('/rag/upload', methods=['POST'])
def rag_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    subject_name = request.form.get('subject_name', 'General')
    unit_number = request.form.get('unit_number', '1')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        if file and file.filename.lower().endswith('.pdf'):
            try:
                file_bytes = file.read()
                
                # Save to MongoDB GridFS
                db = get_db()
                fs = gridfs.GridFS(db)
                file_id = fs.put(file_bytes, filename=file.filename, subject_name=subject_name, unit_number=unit_number)
                
                # For Vercel Serverless, background threads are instantly killed.
                # Must synchronously process the PDF before returning.
                try:
                    chunks = extract_and_chunk_pdf(file_bytes, subject_name, unit_number)
                    if chunks:
                        build_vector_store(chunks)
                    else:
                        print(f"Failed to extract text from {file.filename}")
                except Exception as eval_e:
                    print(f"Processing error: {eval_e}")
                    return jsonify({"error": f"Upload received, but processing failed: {str(eval_e)}. The file might be too complex for the current session."}), 500

                return jsonify({"message": f"Upload complete! File securely saved and vectorized for RAG chat."})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": f"File storage failed: {str(e)}"}), 500
                    
        return jsonify({"error": "Only PDF files are supported"}), 400
    except Exception as global_e:
        traceback.print_exc()
        return jsonify({"error": f"Upload system error: {str(global_e)}"}), 500

@app.route('/rag/comparative_chat', methods=['POST'])
def comparative_chat():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
        
    try:
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Kick off retrieval and plain answer concurrently
            future_context = executor.submit(retrieve_top_chunks, query)
            future_plain_ans = executor.submit(generate_plain_llm_answer, query)
            
            context_chunks = future_context.result()
            
            # Now that context is ready, kick off RAG answer
            future_rag_ans = executor.submit(generate_rag_answer, query, context_chunks)
            plain_ans = future_plain_ans.result()
            rag_ans = future_rag_ans.result()
            
            # Now evaluate both concurrently
            future_rag_score = executor.submit(calculate_faithfulness, rag_ans, context_chunks, query)
            future_plain_score = executor.submit(calculate_general_accuracy, plain_ans, query)
            
            rag_score = future_rag_score.result()
            plain_score = future_plain_score.result()
        
        return jsonify({
            "plain_llm_answer": plain_ans,
            "rag_grounded_answer": rag_ans,
            "faithfulness_score": rag_score,
            "plain_faithfulness_score": plain_score
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/rag/scholar', methods=['POST'])
def rag_scholar_extract():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
        
    try:
        # For scholarships, we increase k for wider search
        context_chunks = retrieve_top_chunks(query) # retrieve_top_chunks uses k=5 by default
        extracted = generate_scholar_info(query, context_chunks)
        
        return jsonify({
            "extracted_info": extracted,
            "chunks_scanned": len(context_chunks)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/scholar/live', methods=['POST'])
def scholar_live():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    try:
        result = fetch_live_opportunities(query, 'scholarship')
        return jsonify({"result": result, "source": "Gemini AI Knowledge Base"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/intern/live', methods=['POST'])
def intern_live():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    from rag_engine import fetch_live_opportunities
    try:
        result = fetch_live_opportunities(query, 'internship')
        return jsonify({"result": result, "source": "Gemini AI Knowledge Base"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# 6. Learn Module Endpoints
# ================================
@app.route('/learn/youtube', methods=['POST'])
def learn_youtube():
    data = request.json
    topic = data.get('topic')
    language = data.get('language', 'English')
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    from rag_engine import fetch_learn_resources
    try:
        result = fetch_learn_resources(topic, 'youtube', language)
        return jsonify({"result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/learn/resources', methods=['POST'])
def learn_resources():
    data = request.json
    topic = data.get('topic')
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    from rag_engine import fetch_learn_resources
    try:
        result = fetch_learn_resources(topic, 'online')
        return jsonify({"result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/learn/papers', methods=['POST'])
def learn_papers():
    data = request.json
    topic = data.get('topic')
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    from rag_engine import fetch_arxiv_papers
    try:
        result = fetch_arxiv_papers(topic)
        return jsonify({"result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

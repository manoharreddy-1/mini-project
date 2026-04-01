import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from database import get_db, get_db_client, DB_NAME
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ── Lazy-loaded AI components ──────────────────────────────────────────────
# We do NOT initialize these at import time because GoogleGenerativeAIEmbeddings
# makes a blocking network call that hangs Flask startup.
# Call _ensure_init() inside any function that needs these.

keys_str = os.getenv("GEMINI_API_KEYS", "")
if keys_str:
    API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
else:
    API_KEYS = [os.getenv("GEMINI_API_KEY")] if os.getenv("GEMINI_API_KEY") else []

# Fallback model chain tried in order when the current model is overloaded (503)
FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

current_key_idx = 0
current_model_idx = 0

def get_current_api_key():
    if not API_KEYS:
        return None
    return API_KEYS[current_key_idx]

def get_current_model():
    return FALLBACK_MODELS[current_model_idx % len(FALLBACK_MODELS)]

def rotate_api_key():
    """Rotate to next API key AND next model on overload."""
    global current_key_idx, current_model_idx, _ai_ready
    current_key_idx = (current_key_idx + 1) % len(API_KEYS) if len(API_KEYS) > 1 else 0
    current_model_idx = (current_model_idx + 1) % len(FALLBACK_MODELS)
    print(f"[Rotated] Switched to Key #{current_key_idx + 1} / Model: {get_current_model()}")
    _ai_ready = False  # Force re-initialization lazily on next call

_embeddings   = None
_llm          = None
gemini_model  = None   # kept at module level for direct SDK use
_ai_ready     = False

def _ensure_init():
    """Initialize AI components on first use (lazy loading)."""
    global _embeddings, _llm, gemini_model, _ai_ready
    if _ai_ready:
        return
    
    key = get_current_api_key()
    if not key:
        print("Warning: No GEMINI_API_KEY found.")
        return

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        import google.generativeai as genai

        model_name = get_current_model()
        genai.configure(api_key=key)
        gemini_model  = genai.GenerativeModel(model_name)
        _embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _llm          = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=key,
                            max_retries=0)
        _ai_ready     = True
        print(f"AI Engines initialized. Key #{current_key_idx + 1}, Model: {model_name}")
    except Exception as e:
        # DO NOT raise - log and leave _ai_ready=False so the caller can handle it
        print(f"Error initializing Google AI Engines (Key #{current_key_idx + 1}, Model: {get_current_model()}): {e}")

# Convenience accessors used by app.py
def get_embeddings():
    _ensure_init()
    return _embeddings

def get_llm():
    _ensure_init()
    return _llm

def get_gemini_model():
    _ensure_init()
    return gemini_model



# ================================
# 1. Ingestion & Chunking
# ================================
def extract_and_chunk_pdf(file_bytes, subject_name, unit_number):
    try:
        doc_fitz = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        print(f"Error reading PDF bytes: {e}")
        return []

    documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=600,
        length_function=len
    )
    
    try:
        for page_num, page in enumerate(doc_fitz):
            try:
                text = page.get_text()
                if text:
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        documents.append({
                            "text": chunk,
                            "metadata": {
                                "subject_name": subject_name,
                                "unit_number": str(unit_number),
                                "page_number": str(page_num + 1)
                            }
                        })
            except Exception as page_e:
                print(f"Error extracting text from page {page_num+1} of {pdf_path}: {page_e}")
                continue
    except Exception as e:
        print(f"Error processing PDF: {e}")
    finally:
        try:
            doc_fitz.close()
        except:
            pass
        
    return documents

def build_vector_store(documents):
    if not documents:
        return
    
    texts = [doc['text'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]
    
    max_attempts = len(API_KEYS) if API_KEYS else 1
    
    for attempt in range(max_attempts):
        _ensure_init()
        try:
            db_client = get_db_client()
            collection = db_client[DB_NAME]["vector_store"]
            
            # Batch ingestion to avoid limits on large files
            batch_size = 150
            total_chunks = len(texts)
            
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=_embeddings,
                index_name="default",
            )
            
            for i in range(0, total_chunks, batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                vector_store.add_texts(texts=batch_texts, metadatas=batch_metas)
                print(f"Ingested batch {i//batch_size + 1}/{ (total_chunks - 1)//batch_size + 1 }")
                
            print(f"Successfully vectorized and stored {total_chunks} chunks into MongoDB Atlas.")
            return
        except Exception as e:
            err_msg = str(e)
            print(f"Ingestion error (attempt {attempt+1}/{max_attempts}): {type(e).__name__}: {str(e)}")
            rotate_api_key()
    
    raise Exception("PDF ingestion failed after trying all API keys/models. Please try again shortly.")

def get_retriever(subject_name=None, unit_number=None):
    _ensure_init()
    db_client = get_db_client()
    collection = db_client[DB_NAME]["vector_store"]
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=_embeddings,
        index_name="default",
    )
    
    search_kwargs = {"k": 5}
    pre_filter = {}
    if subject_name:
        pre_filter["subject_name"] = {"$eq": subject_name}
    if unit_number:
        pre_filter["unit_number"] = {"$eq": str(unit_number)}
        
    if pre_filter:
        search_kwargs["pre_filter"] = pre_filter
        
    return vector_store.as_retriever(search_kwargs=search_kwargs)

def retrieve_top_chunks(query, subject_name=None, unit_number=None):
    try:
        retriever = get_retriever(subject_name, unit_number)
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        print(f"Retrieval error (likely empty DB or embedding issue): {e}")
        return []

# ================================
# 2. Generation (Comparative Mode)
# ================================
def generate_plain_llm_answer(query):
    max_attempts = len(API_KEYS) * len(FALLBACK_MODELS)
    for attempt in range(max_attempts):
        _ensure_init()
        try:
            if gemini_model is None:
                raise Exception("Model not initialized")
            prompt_text = f"""Answer the following academic question in a well-structured format.

Formatting Rules:
- Use **Bold Headings** for each major section (e.g., **Definition**, **Key Points**, **Example**)
- Use bullet points (- ) for listing concepts and details
- **Bold** any important terms or key phrases within sentences
- End with a **Summary** section recapping the main idea in 2-3 bullet points
- Do NOT write in plain paragraphs; always use structured headings and bullets

Question: {query}
Answer:"""
            resp = gemini_model.generate_content(prompt_text, request_options={"timeout": 60})
            return resp.text
        except Exception as e:
            print(f"Plain gen error (attempt {attempt+1}/{max_attempts}): {type(e).__name__}: {str(e)[:80]}")
            rotate_api_key()
    
    return "Generation failed. Gemini API is currently overloaded. Please try again shortly."

def generate_rag_answer(query, retrieved_chunks):
    context_text = ""
    for doc in retrieved_chunks:
        unit = doc.metadata.get('unit_number', 'Unknown')
        context_text += f"[Unit {unit}] {doc.page_content}\n\n"
    
    max_attempts = len(API_KEYS) * len(FALLBACK_MODELS)
    for attempt in range(max_attempts):
        _ensure_init()
        try:
            if gemini_model is None:
                raise Exception("Model not initialized")
            prompt_text = f"""You are a Semester Tutor. Use the provided context to answer the student's question.

Formatting Rules (STRICTLY FOLLOW):
- Use **Bold Headings** for each section (e.g., **Definition**, **Explanation**, **Example**, **From Your Notes**)
- Use bullet points (- ) for all key information and concepts
- **Bold** any critical terms or important keywords inline
- If citing from notes, use: **[Unit X]** before the point
- End with a **Key Takeaways** section with 2-4 bullet points
- Never write in plain paragraph blocks — always use headings + bullets

Constraints:
1. If the context contains the answer, use it and cite [Unit Number].
2. If context is missing info, say "**Not found in your notes**" and explain generally.
3. Highlight where the standard LLM might differ from your notes.

Context:
{context_text}

Question: {query}
Answer:"""
            resp = gemini_model.generate_content(prompt_text, request_options={"timeout": 60})
            return resp.text
        except Exception as e:
            print(f"RAG gen error (attempt {attempt+1}/{max_attempts}): {type(e).__name__}: {str(e)[:80]}")
            rotate_api_key()
    
    return "Generation failed. Gemini API is currently overloaded. Please try again shortly."

# ================================
# 3. Faithfulness Score (LLM-as-a-Judge approach mimicking RAGAS)
# ================================
def calculate_faithfulness(rag_answer, context_chunks, query):
    if not context_chunks:
        return calculate_general_accuracy(rag_answer, query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_chunks])
    
    max_attempts = len(API_KEYS) * len(FALLBACK_MODELS)
    for attempt in range(max_attempts):
        _ensure_init()
        try:
            if gemini_model is None:
                return 0.5
            eval_prompt = f"""You are evaluating a RAG AI response for General Accuracy and Context Grounding.
Given the Context and the Answer, score how objectively accurate and helpful the Answer is.
* If the Answer is both factually accurate and strongly supported by the Context, score 0.9 to 1.0.
* If the Answer is factually accurate but relies mostly on general knowledge because Context is missing/poor, score 0.8 to 0.9.
* If the Answer represents hallucinated or contradicted information, score 0.0 to 0.5.
Analyze carefully, but ONLY RETURN ONE NUMBER LIKE 0.9 - NO OTHER TEXT.

Context: {context_text}
Question: {query}
Answer: {rag_answer}

Score:"""
            resp = gemini_model.generate_content(eval_prompt, request_options={"timeout": 60})
            import re
            match = re.search(r'[01]?\.\d+|1\.0|0\.0', resp.text.strip())
            return float(match.group()) if match else 0.5
        except Exception as e:
            print(f"Faithfulness error (attempt {attempt+1}/{max_attempts}): {str(e)[:80]}")
            rotate_api_key()
    
    return 0.5

def calculate_general_accuracy(answer, query):
    max_attempts = len(API_KEYS) * len(FALLBACK_MODELS)
    for attempt in range(max_attempts):
        _ensure_init()
        try:
            if gemini_model is None:
                return 0.5
            eval_prompt = f"""You are evaluating an AI response for general factual Accuracy.
Score how objectively correct and comprehensive the Answer is for the Question based on general computer science and academic knowledge.
Score: 1.0 = Fully correct, 0.5 = Partially correct/incomplete, 0.0 = Factually incorrect or completely irrelevant.
Analyze carefully, but ONLY RETURN ONE NUMBER LIKE 0.8 - NO OTHER TEXT.

Question: {query}
Answer: {answer}

Score:"""
            resp = gemini_model.generate_content(eval_prompt, request_options={"timeout": 60})
            import re
            match = re.search(r'[01]?\.\d+|1\.0|0\.0', resp.text.strip())
            return float(match.group()) if match else 0.5
        except Exception as e:
            print(f"Accuracy error (attempt {attempt+1}/{max_attempts}): {str(e)[:80]}")
            rotate_api_key()
    
    return 0.5

def generate_quiz(topic: str, level: str, qty: int) -> str:
    """
    Generates a multiple choice quiz on a given topic using Gemini 1.5 Flash.
    Returns a strict JSON string.
    """
    _ensure_init()
    prompt = f"""You are an expert academic tutor.
Generate a multiple choice quiz about the topic: "{topic}".
Difficulty level: {level}
Number of questions: {qty}

You MUST return the output as a RAW JSON array of objects. 
Each object must follow exactly this structure:
{{
  "question": "The question text",
  "options": ["A) first option", "B) second option", "C) third option", "D) fourth option"],
  "correct_answer": "The EXACT string of the correct option from the options list",
  "explanation": "A short, educational explanation of why this answer is correct."
}}

Respond ONLY with the raw JSON array. Do not include any other text or explanation.
"""
    try:
        from google.api_core import retry
        res = gemini_model.generate_content(
            prompt,
            request_options={"retry": retry.Retry(initial=1, maximum=1, timeout=5)}
        )
        text = res.text.strip()
        print(f"DEBUG: Raw AI Response for Quiz: {text[:100]}...") # Log first 100 chars
        
        # More robust JSON extraction
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        # If it's still containing [ and ], try to find the bounds
        if "[" in text and "]" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            text = text[start:end]
            
        return text.strip()
    except Exception as e:
        import traceback
        print(f"Quiz Generation Error: {e}")
        traceback.print_exc()
        return "[]"

def generate_scholar_info(query, retrieved_chunks):
    _ensure_init()
    if not retrieved_chunks:
        return "No relevant context found in your documents to answer this query."

    context_text = "\n".join([f"--- Context Segment ---\n{doc.page_content}" for doc in retrieved_chunks])
    
    prompt = f"""You are a Scholarship & Internship Scout. 
Extract any relevant information about scholarships, internships, or academic opportunities matching the user's query from the context provided.

Context:
{context_text}

User Query: {query}

Formatting Rules:
- Use **Bold Headings** for each opportunity name
- Use bullet points for eligibility, award amount, deadline, and application link
- **Bold** key terms like **Eligibility**, **Award**, **Deadline**, **How to Apply**
- End with a **Summary** section
- If no info found, say: "No specific matches found for '{query}' in uploaded materials."

Answer:"""
    
    try:
        from google.api_core import retry
        response = gemini_model.generate_content(
            prompt,
            request_options={"retry": retry.Retry(initial=1, maximum=1, timeout=5)}
        )
        if response and response.text:
            return response.text
        return "AI returned an empty response. Please try rephrasing your query."
    except Exception as e:
        print(f"Scholar extraction error (SDK): {e}")
        return f"Error extracting information via AI: {str(e)}"


def fetch_live_opportunities(query: str, opp_type: str) -> str:
    """
    Uses Gemini's knowledge to generate fresh, structured scholarship or internship
    listings for the given query, formatted in rich markdown.
    opp_type: 'scholarship' or 'internship'
    """
    if opp_type == 'scholarship':
        detail_fields = "**Eligibility**, **Award Amount**, **Deadline / Typical Timeline**"
        how_to_apply = "**Official Application Link:** [Must provide a real, clickable markdown link to the official site, e.g. `[Apply Here](https://...)`]"
    else:
        detail_fields = "**Role**, **Typical Stipend**, **Duration**, **Skills Required**"
        how_to_apply = "**Official Application Link:** [Must provide a real, clickable markdown link to the official company careers page or portal, e.g. `[Apply Here](https://...)`]"

    prompt = f"""You are an expert career and {opp_type} advisor for students.

A student searched for: "{query}"

List 5 highly accurate, well-known {opp_type} opportunities matching this search based on your current knowledge of the world. Provide real programs.

Format EACH opportunity exactly like this:

### [Number]. **[Opportunity Name]**
- {detail_fields}
- {how_to_apply}
- **Who Can Apply:** [one line eligibility]

After all 5, add:

---
### 💡 Quick Tips
- [2 actionable tips for applying to these {opp_type}s]

Rules: Provide the most accurate and up-to-date information you have. Use headings and bullets only — no plain paragraphs. Do NOT hallucinate fake URLs; use the closest real official URL you know of for the organization. ALWAYS format the URL as a markdown link.
"""

    max_attempts = len(API_KEYS) if API_KEYS else 1
    attempts = 0

    while attempts < max_attempts:
        current_key = get_current_api_key()
        if not current_key:
            return "Error: No API key configured."

        import google.generativeai as genai
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel(get_current_model())

        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
            return "AI returned an empty response. Please try a different search term."
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                print(f"Quota exceeded for Key #{current_key_idx + 1}, trying next key...")
                rotate_api_key()
                attempts += 1
            else:
                print(f"[fetch_live_opportunities] Gemini error: {e}")
                return f"Error generating results: {str(e)}"
    
    return "Error: All available API keys have exceeded their quota. Please configure billing or wait for the quota to reset."

# ================================
# 5. Learn Module (YouTube, Resources, Papers)
# ================================
def fetch_learn_resources(topic: str, mode: str, language: str = "English") -> str:
    """
    Fetches learning resources (YouTube or Online Resources) using Gemini.
    """
    if mode == 'youtube':
        import urllib.request
        import urllib.parse
        import re
        search_query = f"{topic} tutorial course in {language}"
        search_url = 'https://www.youtube.com/results?search_query=' + urllib.parse.quote(search_query)
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        context_videos = ""
        try:
            html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
            video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
            seen = set()
            unique_ids = [x for x in video_ids if not (x in seen or seen.add(x))][:5]
            for i, vid in enumerate(unique_ids):
                link = f"https://www.youtube.com/watch?v={vid}"
                thumb = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
                context_videos += f"\n{i+1}. Video URL: {link}\nImage Thumbnail: {thumb}\n"
        except Exception as e:
            print(f"YouTube search error: {e}")
        
        prompt = f"""You are an educational tutor finding the best video tutorials.
Topic: {topic}
Language: {language}

Below are the top 5 actual live YouTube working links and thumbnails we just fetched:
{context_videos}

Write a helpful markdown guide presenting exactly these 5 videos to the user.
For EACH video:
1. Provide a "Why this?" explanation detailing why it's a good choice.
2. Embed the EXACT Image URL provided as a markdown image linking to the Video URL.
Format example: [![Video Title](Image URL)](Video URL)

Format exactly like this:
### 📺 Top 5 YouTube Videos in {language}
1. **[Video Title](Video URL)** by Channel Name
   [![Video Title](Image URL)](Video URL)
   - **Why this?** [Your Explanation]

(Continue for all 5 videos)
"""
    elif mode == 'online':
        prompt = f"""You are an educational tutor finding the best text-based courses.
Recommend the Top 5 best free online text resources/platforms to learn: "{topic}".

Prioritize high-quality written tutorials like GeeksforGeeks, W3Schools, MDN Web Docs, Tutorialspoint, or official documentation.

CRITICAL INSTRUCTION:
You MUST provide actual, clickable URLs for these resources using Markdown format: `[Platform Name](https://www.actual-url.com/...)`.
Do NOT just tell the user to Google it. Give them the most accurate direct link you know to the tutorial.

Format exactly like this:
### 🌐 Top 5 Online Text Resources
1. **[Platform/Resource Name]** - [Topic Specific Guide]
   - **Link:** [Clickable Markdown Link to the resource]
   - **Why this?** [Short reason]
2. ... (continue for 5 resources)
"""
    else:
        return "Invalid mode selected."

    max_attempts = len(API_KEYS) if API_KEYS else 1
    attempts = 0

    while attempts < max_attempts:
        current_key = get_current_api_key()
        if not current_key:
            return "Error: No API key configured."

        import google.generativeai as genai
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel(get_current_model())

        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
            return "AI returned an empty response. Please try a different topic."
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                rotate_api_key()
                attempts += 1
            else:
                return f"Error generating results: {str(e)}"
    
    return "Error: All available API keys have exceeded their quota."

def fetch_arxiv_papers(topic: str) -> str:
    """
    Fetches 10 real, free research papers on the given topic from arXiv via API.
    """
    try:
        import arxiv
        # Construct the default API client.
        client = arxiv.Client()
        
        # Search for the top 10 papers related to the topic
        search = arxiv.Search(
            query = topic,
            max_results = 10,
            sort_by = arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))
        
        if not results:
            return f"### 📄 Research Papers on '{topic}'\nNo recent papers found on arXiv. Try a broader topic."

        markdown_output = f"### 📄 Top {len(results)} Free Research Papers on '{topic}'\n\n"
        markdown_output += "*All papers listed below are free & open-access via arXiv.*\n\n"
        
        for i, paper in enumerate(results, start=1):
            authors = ", ".join([author.name for author in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."
                
            published_date = paper.published.strftime("%B %Y")
            
            markdown_output += f"{i}. **[{paper.title}]({paper.pdf_url})**\n"
            markdown_output += f"   - **Authors:** {authors}\n"
            markdown_output += f"   - **Published:** {published_date}\n"
            markdown_output += f"   - **Read Free PDF:** [Click Here]({paper.pdf_url})\n\n"
            
        return markdown_output
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error fetching research papers: {str(e)}\nPlease try again later."

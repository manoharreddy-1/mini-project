import os
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
keys_str = os.getenv("GEMINI_API_KEYS", "")
if keys_str:
    API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
else:
    API_KEYS = [os.getenv("GEMINI_API_KEY")] if os.getenv("GEMINI_API_KEY") else []

if not API_KEYS:
    print("Error: No API keys found.")
    exit(1)

genai.configure(api_key=API_KEYS[0])

# Evaluation Dataset for Comparitive Study
eval_data = [
    {
        "query": "What are the main components of a RAG system?",
        "context": "A Retrieval-Augmented Generation (RAG) system mainly consists of two components: a retriever that fetches relevant documents from a knowledge base, and a generator (usually an LLM) that synthesizes the final answer using the retrieved context.",
        "ground_truth": "The main components are a retriever and a generator."
    },
    {
        "query": "How does an embedding vector store work?",
        "context": "An embedding vector store stores mathematical representations (vectors) of text. When a query is made, it is also converted into a vector, and the store finds the closest vectors using similarity metrics like cosine similarity.",
        "ground_truth": "It stores text as mathematical vectors and uses similarity metrics like cosine similarity to find the closest matches to a query vector."
    },
    {
        "query": "What is the role of ChromaDB in LangChain?",
        "context": "ChromaDB is an open-source vector database. In LangChain, it is often used as a vector store to store and query embeddings efficiently for RAG applications.",
        "ground_truth": "ChromaDB acts as an open-source vector store to efficiently store and query embeddings."
    },
    {
        "query": "Why do RAG models sometimes hallucinate less than standard LLMs?",
        "context": "RAG models reduce hallucinations because they are grounded in provided external knowledge, forcing the LLM generation to rely on retrieved facts rather than solely on its parametric memory.",
        "ground_truth": "They are grounded in retrieved external knowledge, which helps prevent relying solely on parametric memory."
    },
    {
        "query": "What is a chunk overlap in text splitting?",
        "context": "Chunk overlap is a technique used in text splitting where consecutive chunks share a specified number of characters or tokens. This ensures that context is not lost at the boundaries of chunks.",
        "ground_truth": "It's when consecutive text chunks share tokens to preserve context across boundaries."
    }
]

def evaluate_answer_accuracy(answer, ground_truth, query):
    prompt = f"""You are evaluating an AI response for Accuracy against a known Ground Truth.
Score how objectively accurate the Answer is compared to the Ground Truth.
Score: 1.0 = Fully correct and covers all ground truth facts, 0.5 = Partially correct, 0.0 = Incorrect or entirely missed the facts.
ONLY RETURN ONE NUMBER LIKE 0.8 - NO OTHER TEXT.

Question: {query}
Ground Truth: {ground_truth}
Answer: {answer}

Score:"""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(prompt)
        import re
        match = re.search(r'[01]?\.\d+|1\.0|0\.0', resp.text.strip())
        return float(match.group()) if match else 0.5
    except Exception as e:
        print(f"Eval error: {e}")
        return 0.5

models_to_evaluate = [
    {"name": "Standard LLM Baseline", "is_rag": False, "model": "gemini-2.5-flash"},
    {"name": "RAG Model 1 (Retriever + gemini-2.5-flash)", "is_rag": True, "model": "gemini-2.5-flash"},
    {"name": "RAG Model 2 (Retriever + gemini-2.0-flash)", "is_rag": True, "model": "gemini-2.0-flash"},
    {"name": "RAG Model 3 (Retriever + gemini-2.5-pro)", "is_rag": True, "model": "gemini-2.5-pro"},
    {"name": "RAG Model 4 (Retriever + gemma-3-12b-it)", "is_rag": True, "model": "gemma-3-12b-it"},
    {"name": "RAG Model 5 (Retriever + gemini-flash-latest)", "is_rag": True, "model": "gemini-flash-latest"}
]

results = []

print("Starting Comparative Evaluation...")

for m in models_to_evaluate:
    print(f"Evaluating {m['name']}...")
    model_instance = genai.GenerativeModel(m['model'])
    
    total_score = 0.0
    for i, item in enumerate(eval_data):
        try:
            if m["is_rag"]:
                prompt = f"Use the provided context to answer the question.\nContext: {item['context']}\nQuestion: {item['query']}\nAnswer:"
            else:
                prompt = f"Answer the following question based on your own knowledge.\nQuestion: {item['query']}\nAnswer:"
            
            resp = model_instance.generate_content(prompt)
            answer = resp.text
            
            score = evaluate_answer_accuracy(answer, item["ground_truth"], item["query"])
            total_score += score
            time.sleep(2) # rate limit to avoid 429
        except Exception as e:
            print(f"Error generating for {m['name']} on Q{i+1}: {e}")
            score = 0.0
            
    avg_accuracy = (total_score / len(eval_data)) * 100
    results.append({
        "Model Configuration": m["name"],
        "Type": "RAG Pipeline" if m["is_rag"] else "Standard LLM",
        "Base Model": m["model"],
        "Average Accuracy (%)": f"{avg_accuracy:.2f}%"
    })
    print(f"  -> Accuracy: {avg_accuracy:.2f}%\n")

df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(df.to_markdown(index=False))
df.to_csv("rag_comparative_results.csv", index=False)

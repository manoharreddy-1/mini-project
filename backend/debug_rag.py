import sys
import traceback
sys.path.append('.')
from rag_engine import retrieve_top_chunks, generate_rag_answer

try:
    chunks = retrieve_top_chunks('What is machine learning?')
    print(f"Chunks retrieved: {len(chunks)}")
    ans = generate_rag_answer('What is machine learning?', chunks)
    print(ans)
except Exception as e:
    print("--- ERROR DUMP ---")
    traceback.print_exc()

import os
from rag_engine import generate_quiz
from dotenv import load_dotenv

load_dotenv()

topic = "Python Programming"
level = "Medium"
qty = 3

print(f"Generating quiz for topic: {topic}", flush=True)
result = generate_quiz(topic, level, qty)
print("--- RAW RESULT ---", flush=True)
print(result, flush=True)
print("--- END RAW RESULT ---", flush=True)

import json
try:
    questions = json.loads(result)
    print(f"Successfully parsed {len(questions)} questions.", flush=True)
except Exception as e:
    print(f"Failed to parse JSON: {e}", flush=True)

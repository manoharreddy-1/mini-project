import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Listing ALL models...", flush=True)
try:
    models = genai.list_models()
    for m in models:
        print(f"Name: {m.name}, Methods: {m.supported_generation_methods}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

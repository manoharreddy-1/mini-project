import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Listing models one by one...", flush=True)
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            name = m.name.replace('models/', '')
            print(f"FOUND_MODEL: {name}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

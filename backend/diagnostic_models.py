import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

keys_str = os.getenv('GEMINI_API_KEYS', os.getenv('GEMINI_API_KEY', ''))
API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]

for i, key in enumerate(API_KEYS):
    print(f"\n--- Checking Key {i+1} ({key[:10]}...) ---")
    genai.configure(api_key=key)
    try:
        models = genai.list_models()
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"  Available: {m.name}")
    except Exception as e:
        print(f"  ERROR for Key {i+1}: {e}")

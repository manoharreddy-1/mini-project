import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

models_to_test = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

for m in models_to_test:
    print(f"\n--- Testing {m} ---")
    try:
        model = genai.GenerativeModel(m)
        response = model.generate_content("Hello! Are you working?")
        print(f"SUCCESS! Response: {response.text.strip()}")
    except Exception as e:
        print(f"ERROR: {e}")

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

models_to_try = [
    'gemini-1.5-flash',
    'models/gemini-1.5-flash',
    'gemini-1.5-pro',
    'models/gemini-1.5-pro',
    'gemini-pro',
    'models/gemini-pro'
]

for model_name in models_to_try:
    print(f"Testing model: {model_name}...", flush=True)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'OK'", request_options={"timeout": 10})
        print(f"SUCCESS with {model_name}: {response.text}", flush=True)
        break
    except Exception as e:
        import traceback
        print(f"FAILED with {model_name}:", flush=True)
        traceback.print_exc()

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}", flush=True)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

print("Sending test request to Gemini...", flush=True)
try:
    response = model.generate_content("Say 'Hello World'", request_options={"timeout": 30})
    print(f"Response: {response.text}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

import os
import traceback
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Hello world!")
    print("SUCCESS: ", response.text)
except Exception as e:
    print("FAILED")
    traceback.print_exc()

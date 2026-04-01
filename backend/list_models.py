import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

keys_str = os.getenv('GEMINI_API_KEYS','')
API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]
if not API_KEYS:
    API_KEYS = [os.getenv('GEMINI_API_KEY')]

genai.configure(api_key=API_KEYS[0])
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

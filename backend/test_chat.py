import requests

url = "http://127.0.0.1:5000/chat/message"
data = {
    "session_id": 1,
    "user_id": 1,
    "query": "java inheritance"
}

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
except Exception as e:
    print("Request failed:", e)

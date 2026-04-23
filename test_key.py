import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"API Key found: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"Key length: {len(api_key)}")
    print(f"Key starts with: {api_key[:10]}...")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=api_key)
    res = llm.invoke("Say hello")
    print(f"Response: {res.content}")
except Exception as e:
    print(f"Error: {e}")

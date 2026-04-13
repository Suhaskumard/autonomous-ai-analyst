import os
import sys
from google import genai
from dotenv import load_dotenv

# Load .env from the backend directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def test_connection():
    api_key = os.getenv("GEMINI_API_KEY")
    model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please add it to your .env file.")
        return

    print(f"Testing connection to Gemini with model: {model_id}...")
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents="Say 'Chatbot connection successful!'"
        )
        print(f"Success! Response: {response.text.strip()}")
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        print("\nPossible solutions:")
        print("1. Verify your GEMINI_API_KEY is correct.")
        print("2. Run 'backend/scratch/list_models.py' to see if your key supports the default model.")
        print("3. Check your internet connection.")

if __name__ == "__main__":
    test_connection()

import streamlit as st
import os
from google import genai
from dotenv import load_dotenv

@st.cache_data(ttl=300, show_spinner=False)
def generate_ai_playbook(prompt: str, _prompt_hash=None) -> str:
    try:
        # 1. Initialize the new client with your API key
        load_dotenv()  # Load environment variables from .env file
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # 2. Generate content using the new client.models syntax
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        # 3. Return the generated text
        if response and response.text:
            return response.text
        return "_(No response generated)_"
        
    except Exception as e:
        return f"_(Playbook generation unavailable: {e})_"
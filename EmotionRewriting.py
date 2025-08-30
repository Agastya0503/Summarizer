from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please add it to .env file.")

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Prompt template for Emotion Rewriting
prompt = ChatPromptTemplate.from_template(
    """
You are given a sentence:

"{sentence}"

Rewrite it in THREE different emotional tones:

1. Happy Tone 😀  
2. Sad Tone 😢  
3. Angry Tone 😡  
"""
)

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# === Run in terminal ===
if __name__ == "__main__":
    sentence = input("Enter a sentence:\n")
    result = chain.invoke({"sentence": sentence})
    
    print("\n=== Emotion Rewriting ===")
    print(result["text"])

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please add it to .env file.")

# Initialize the Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Define the prompt template for perspective shift
prompt = ChatPromptTemplate.from_template(
    """
You are given an event:

"{event}"

Summarize this event from TWO different perspectives:

1. Hero’s Perspective: Make the hero sound noble, brave, and positive.  
2. Villain’s Perspective: Make the villain sound justified or misunderstood.
"""
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# === Run in terminal ===
if __name__ == "__main__":
    event = input("Enter an event:\n")
    result = chain.invoke({"event": event})   # invoke is modern replacement
    
    print("\n=== Perspective Shift Summarization ===")
    print(result["text"])

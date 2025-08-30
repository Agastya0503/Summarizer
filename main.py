from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Add it to .env file.")

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are given a research abstract:

"{abstract}"

Summarize it in TWO distinct styles:

1. Scientist Summary: Focus on technical accuracy, scientific terms, and details.  
2. News Reporter Summary: Make it simple, engaging, and public-friendly.
""")

# Use LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    abstract = input("Enter research abstract:\n")
    result = chain.invoke({"abstract": abstract})  # ✅ use invoke()
    print("\n=== Multi-Role Summarization ===")
    print(result["text"])

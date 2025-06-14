# src/llm_config.py
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI as LangChainChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv() # Load environment variables from .env file

def get_chat_model(model_name: str = "deepseek/deepseek-r1:free", temperature: float = 0.7): # <--- UPDATED THIS LINE
    """
    Initializes and returns the Chat LLM using OpenRouter.
    Accepts model_name as a parameter.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables or .env file.")

    return LangChainChatOpenAI(
        model=model_name,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,
        temperature=temperature
    )

def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes and returns the Embeddings model using Hugging Face.
    Accepts model_name as a parameter.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
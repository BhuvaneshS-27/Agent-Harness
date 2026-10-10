import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
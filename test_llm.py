import ollama
import config

client = ollama.Client(host=config.OLLAMA_HOST)
reply = client.chat(
    model=config.LLM_MODEL,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply["message"]["content"])
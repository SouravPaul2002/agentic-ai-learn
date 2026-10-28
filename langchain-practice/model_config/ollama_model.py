from langchain_ollama import ChatOllama

def load_model():
    model = ChatOllama(
        model="qwen2.5-coder:3b",
        temperature=0.3
    )
    return model

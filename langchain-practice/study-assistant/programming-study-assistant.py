from clear_history import clear_history
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (HumanMessage, SystemMessage, AIMessage)
import sys
from pathlib import Path

# Add project root to sys.path so imports work from any subdirectory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model_config.ollama_model import load_model
model = load_model()

parser= StrOutputParser()

template= ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an expert Programming tutor. Explain clearly in simple terms."),
    MessagesPlaceholder("chat_history"),
    ("human","{question}")
])
quiz_template= ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a quiz master and you need to create a Quiz on the given topic"),
    ("human","{topic}")
])

summary_template= ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an expert to summarize a given content. Based on the following content please give a summary in breif"),
    MessagesPlaceholder("chat_history"),("human","Please provide a brief summary of our discussion so far.")
])

chat_history=[]

chain= template | model | parser
summary_chain= summary_template | model | parser
quiz_chain= quiz_template | model | parser

print("Chat started")
while True:

    question= input("You: ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        break

    if question.lower() == "clear":
        clear_history(chat_history)
        print("System: Your chat history has been cleared.")
        continue

    if question.lower() == "summary":
        if not chat_history:
            print("System: No conversation history to summarize.")
            continue
        print("System: Here is your Summary based on the chat History: ")
        

        summary_answer= summary_chain.invoke({
            "chat_history": chat_history
        })

        print(summary_answer)

        chat_history.extend([
            HumanMessage(content="Provide a detailed summary on the conversation so far"),
            AIMessage(content=summary_answer)
        ])
        continue

    if question.lower() == "quiz":
        topic= input("System: Please provide the Quiz topic: ")

        if not topic:
            print("System: Quiz cancelled - Topic cannot be empty.")
            continue

        quiz_answer= quiz_chain.invoke({
            "topic": topic
        })
        print("AI: ",quiz_answer)
        chat_history.extend([
            HumanMessage(content=f"Create a quiz on: {topic}"),
            AIMessage(content=quiz_answer)
        ])
        continue
    
    answer= chain.invoke({
        "chat_history": chat_history,
        "question": question
    })

    print("AI: ",answer)

    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])
print("Chat ended")
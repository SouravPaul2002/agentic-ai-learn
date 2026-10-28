from langchain_core.messages.system import SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_ollama import ChatOllama



load_dotenv()

model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.3
)

template= ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful python tutor. Explain Clearly in simple english"),
    MessagesPlaceholder("chat_history"),
    ("human","{question}")
])

parser= StrOutputParser()

chain= template | model | parser


chat_history=[]

print("AI Chatbot Started. Type exit to stop")
while True:
    question= input("You: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    answer= chain.invoke({
        "chat_history": chat_history,
        "question":question}
        )

    print("AI: ",answer)

    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])
print("Chat Ended")
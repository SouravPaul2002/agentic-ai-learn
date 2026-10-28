# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# load_dotenv()

# model= GoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )

# prompt= ChatPromptTemplate.from_messages([
#     ("system",
#     "You are a programming teacher. Explain clearly"),
#     ("human",
#     "Explain {topic} in exactly 3 bullet points")
# ])

# parser= StrOutputParser()

# chain= prompt | model | parser

# result= chain.invoke({"topic":"Python functions"})

# print(result)




# using chatHistory 

from langchain_core.messages import content
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model = GoogleGenerativeAI(model="gemini-2.5-flash")

chat_history=[
    SystemMessage(
        content="You are an good explainer. Explain clearly"
    ),
    HumanMessage(
        content="Who is Narendra Modi?"
    ),
    AIMessage(
        content="Narendra Modi is the current Prime Minister of India and a prominent leader of the Bharatiya Janata Party (BJP)."
    ),
    HumanMessage(
        content="How old is he?"
    )
]

result= model.invoke(chat_history)
print(result)
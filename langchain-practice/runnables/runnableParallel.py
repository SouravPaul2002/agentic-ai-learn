from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.3
)

prompt1= PromptTemplate(
    template="create a tweet on {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="create a linkedin post on {topic}",
    input_variables=["topic"]
)

parser= StrOutputParser()

chain= RunnableParallel({
    "tweet": RunnableSequence(prompt1, model , parser),
    "linkedin": RunnableSequence(prompt2, model , parser)
})

result= chain.invoke({"topic": "AI"})

print(result)
print("-----------------------------")

print(result["tweet"])

print("-----------------------------")

print(result["linkedin"])
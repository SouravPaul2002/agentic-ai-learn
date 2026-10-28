from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.3
)

prompt1= PromptTemplate(
    template="Create a joke on {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Explain the following joke -> {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain= RunnableSequence(prompt1, model , prompt2 , model , parser)

result= chain.invoke({"topic":"Black hole"})

print(result)
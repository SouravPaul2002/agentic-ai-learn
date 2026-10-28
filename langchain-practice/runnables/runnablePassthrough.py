from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
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
    template="explain the following joke -> {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

joke_print= RunnablePassthrough()

joke_chain= RunnableSequence(prompt1, model , parser)

explain_chain= RunnableParallel({
    "joke": joke_print,
    "explain": RunnableSequence(prompt2, model, parser)
})

final_chain= RunnableSequence(joke_chain, explain_chain)

result= final_chain.invoke({
    "topic": "Langchain"
})

print(result)
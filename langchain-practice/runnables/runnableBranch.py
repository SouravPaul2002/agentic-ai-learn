from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.3
)

prompt1= PromptTemplate(
    template="generate a report on {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="summarize the following text in 1 line only -> {text}",
    input_variables=["text"]
)

parser= StrOutputParser()

report_gen_chain= prompt1 | model | parser
branch_chain= RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain= report_gen_chain | branch_chain
result= final_chain.invoke({
    "topic": "India"
})

print(result)
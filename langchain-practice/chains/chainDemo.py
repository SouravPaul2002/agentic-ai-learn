from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.3
)

parser= StrOutputParser()

report_prompt= ChatPromptTemplate.from_template('''
    Write a detailed report on {topic}
''')

fact_prompt= ChatPromptTemplate.from_template('''
    Give me top 5 facts from the following content: {report}
''')

chain= report_prompt | model | parser | fact_prompt | model | parser

result= chain.invoke({
    "topic":"Black hole"
})

print(result)
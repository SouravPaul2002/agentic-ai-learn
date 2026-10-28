from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.3
)

class Student(BaseModel):
    name: str = Field(description="Name of the student")
    age: int = Field(description="age of the student")
    course: str = Field(description="Course studied by the Student")

parser= PydanticOutputParser(
    pydantic_object=Student
)

prompt= ChatPromptTemplate.from_template(
    '''
    Extract the student details from the text. \n
    {format_instruction}
    student details: {student_text}
    '''
)

prompt= prompt.partial(
    format_instruction=parser.get_format_instructions()
)

chain= prompt | model | parser

student_details= "Sourav paul is 24 year old MCA student."

result = chain.invoke({
    "student_text": student_details
})

print(result)
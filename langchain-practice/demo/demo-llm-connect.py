from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt= PromptTemplate.from_template(
    '''
    Explain me {topic} in {language} in a {style} way in 2-3 lines
    '''
)

formatted_prompt=prompt.format(
    topic="Cricket",
    language="english",
    style="creative"
)

result= model.invoke(formatted_prompt)

print(result.content)
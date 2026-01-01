from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str =Field(description="Name of the person")
    age:int = Field(gt=18,description="Age of the person")
    city:str = Field(description="Name of the city the person resides in")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="Generate the name,age,city of a fictional {nationality} person. \n {format_instructions}",
    input_variables=["nationality"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain= template | model | parser
final_res= chain.invoke({"nationality":"Japanese"})

# prompt=template.invoke({"nationality":"British"})

# res=model.invoke(prompt)

# final_res=parser.parse(res.content)

print(final_res)

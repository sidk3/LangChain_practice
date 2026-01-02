from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.7,
    max_new_tokens=200
)

prompt=PromptTemplate(
    template="Generate 3 interesting facts about {topic}",
    input_variables=['topic']
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

chain=prompt | model | parser

res=chain.invoke({"topic":"Football"})

print(res)
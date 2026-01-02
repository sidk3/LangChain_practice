from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.7,
    max_new_tokens=200
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate a brief summary of the following text.\n {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

chain= prompt1 | model | parser | prompt2 | model | parser

res=chain.invoke({"topic":"MCP Servers"})

print(res)

#To print the sequence of steps
#chain.get_graph().print_ascii()
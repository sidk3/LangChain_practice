from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0,
    max_new_tokens=120
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Give me a joke about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Explain the below joke: {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({"topic":"AI"}))

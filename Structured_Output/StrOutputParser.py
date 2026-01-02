from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

temp1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

temp2=PromptTemplate(
    template="Write a 5 line summary of following text \n{text}",
    input_variables=['text']
)

prompt1=temp1.invoke({'topic':'black hole'})
res=model.invoke(prompt1)
prompt2=temp2.invoke({'text':res.content})
#parser=StrOutputParser()
res1=model.invoke(prompt2)
#chain=temp1 | model | parser | temp2 | model | parser

print(res1.content)
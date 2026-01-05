from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableBranch

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.1,
    max_new_tokens=150
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Give a summary of following text.\n {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

repo_gen_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(repo_gen_chain,branch_chain)

print(final_chain.invoke({"topic":"Global warming"}))
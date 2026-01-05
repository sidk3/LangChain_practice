from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0,
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "Word_Count":RunnableLambda(word_count)
})

# Alternative: 
# parallel_chain=RunnableParallel({
#     "joke":RunnablePassthrough,
#     "Word_Count":RunnableLambda(lambda x: len(x.split()))
# })

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

res=final_chain.invoke({"topic":"AI Agents"})

print(res)

final_res="""{} \n word count - {}""".format(res["joke"],res["Word_Count"])

print(final_res)
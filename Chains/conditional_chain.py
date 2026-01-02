from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0,
)

model=ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Provide the sentiment of the feedback.")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="Classify the sentiment of the below feedback into Positive or Negative only.\n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

parser1= StrOutputParser()

classifier_chain= prompt1 | model | parser2

prompt2=PromptTemplate(
    template="Generate appropriate response for the below positive feedback. \n {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="Generate appropriate response for the below negative feedback. \n {feedback}",
    input_variables=["feedback"]
)

#res= classifier_chain.invoke({"feedback":"I had a really bad experience with this product."}).sentiment

branch_chain=RunnableBranch(
    (lambda x: x.sentiment=="Positive", prompt2 | model | parser1),
    (lambda x: x.sentiment=="Negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

chain= classifier_chain | branch_chain

res= chain.invoke({"feedback": "I had a really bad experience with this product."})

print(res)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from typing import TypedDict

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.7,
    max_new_tokens=200
)

chat = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary:str
    sentiment:str

structured_model=chat.with_structured_output(Review)
response = structured_model.invoke([
    HumanMessage(content="The hardware is not working as expected. I have tried restarting it multiple times but the issue persists. The UI also seems to be unresponsive at times. Hope a new update can fix these issues.")
])

print(response)


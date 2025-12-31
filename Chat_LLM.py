from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
import streamlit as st

st.header("LangChain assistant")
user_input=st.text_input("Enter your prompt: ")


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.7,
    max_new_tokens=200
)


chat = ChatHuggingFace(llm=llm)

response = chat.invoke([
    HumanMessage(content=user_input)
])

if st.button("Summarise"):
    st.write(response.content)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st

st.header("LangChain assistant")

player=st.selectbox("Choose Player",["Novak Djokovic","Roger Federer","Rafael Nadal"])
tournament=st.selectbox("Choose tournament",["Wimbledon","US Open","French Open"])
length=st.selectbox("Choose length",["Short(1-2 paragraphs)","Medium(3-4 paragraphs)","Long(5-6 paragraphs)"])

template=PromptTemplate(
    template="""
    You are a sports journalist, write a detailed article about {player} performance in {tournament}.
    The article should be of {length} length.
""",
input_variables=['player','tournament','length']
)

prompt=template.invoke(
    {
        'player':player,
        'tournament':tournament,
        'length':length
    }
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.7,
    max_new_tokens=200
)

chat = ChatHuggingFace(llm=llm)

if st.button("Summarise"):
    response = chat.invoke([
    HumanMessage(content=prompt.to_string())
])
    st.write(response.content)
#print(response.content)

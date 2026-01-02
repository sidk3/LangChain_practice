from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

prompt1= PromptTemplate(
    template="Generate notes based on the below text.\n {text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="Generate a 3 short questions and answers based on the below text.\n {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz.\n notes => {notes} and quiz => {quiz}",
    input_variables=['notes','quiz']
)
llm1=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.7,
)

model1=ChatHuggingFace(llm=llm1)

llm2=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model2=ChatHuggingFace(llm=llm2)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain= prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text= """
Artificial Intelligence (AI) Agents

An AI agent is an autonomous software entity that perceives its environment, makes decisions, and performs actions to achieve specific goals. It continuously interacts with its surroundings through sensors (inputs) and actuators (outputs), using reasoning or learned knowledge to choose the best possible action at each step.

At the core of an AI agent is the agent function, which maps percepts (what the agent observes) to actions. This function can be implemented using simple rule-based logic, search algorithms, probabilistic reasoning, or machine learning models such as reinforcement learning. More advanced agents are capable of learning from past experiences and improving their performance over time.

AI agents can be reactive, responding directly to current inputs, or deliberative, where they maintain an internal model of the world and plan ahead. In real-world systems, many agents are hybrid, combining fast reactions with long-term planning. Multi-agent systems involve multiple AI agents that cooperate or compete to solve complex problems.

AI agents are widely used in applications such as virtual assistants, recommendation systems, autonomous vehicles, robotic process automation, game AI, and decision-support systems. Their key strengths lie in autonomy, adaptability, and the ability to operate in dynamic and uncertain environments
"""
res=chain.invoke({'text':text})

print(res)

print(chain.get_graph().print_ascii())

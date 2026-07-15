# LangChain Practice

A hands-on repository for learning and experimenting with **LangChain**, covering everything from prompt engineering to advanced topics like **RAG (Retrieval-Augmented Generation)**, **semantic search**, and **structured outputs**.

Each module focuses on a specific LangChain concept with practical, easy-to-understand examples.

---

## Features

- Prompt Templates and Chat Models
- LangChain Expression Language (LCEL)
- Chains and Runnables
- Structured Outputs with Pydantic
- Semantic Search using Embeddings
- Retrieval-Augmented Generation (RAG)
- HuggingFace Model Integration
- YouTube Q&A Bot

---

## Repository Structure

```
LangChain_practice/
├── Chains/
├── Chat_LLM_demo/
├── Prompts/
├── Runnables/
├── Structured_Output/
├── Semantic_Search/
├── RAG/
├── requirements.txt
└── README.md
```

### Module Overview

#### Chains
Build reusable pipelines by combining prompts, LLMs, and output parsers.

#### Prompts
Learn prompt templates, dynamic variables, and prompt engineering techniques.

#### Chat_LLM_demo
Basic interaction with chat-based language models.

#### Runnables
Explore the Runnable interface, including:
- Sequential execution
- Parallel execution
- Branching
- Data transformations

#### Structured_Output
Generate validated outputs using **Pydantic** models.

#### Semantic_Search
Implement embedding-based similarity search using vector stores.

#### RAG
Build Retrieval-Augmented Generation pipelines, including a YouTube transcript Q&A bot.

---

## Tech Stack

- Python
- LangChain
- HuggingFace Transformers
- Sentence Transformers
- FAISS / Chroma
- Pydantic

---

## Installation

```bash
git clone https://github.com/<your-username>/LangChain_practice.git
cd LangChain_practice
pip install -r requirements.txt
```

---

## Key Learnings

Through this repository, you'll gain practical experience with:

- Prompt Engineering using Prompt Templates
- Working with Chat Models and LLMs
- LangChain Expression Language (LCEL)
- Building Sequential, Parallel, and Conditional Chains
- Runnable Interface and Data Flow
- Output Parsing and Structured Responses with Pydantic
- Semantic Search using Embeddings and Vector Stores
- Retrieval-Augmented Generation (RAG)
- Integrating HuggingFace models with LangChain
- Building end-to-end AI applications with retrieval pipelines

---

## Best Practices Demonstrated

- Modular LangChain workflows
- Reusable prompt templates
- LCEL-based pipeline composition
- Structured output validation
- Retrieval-first architecture for RAG
- Clean separation of prompts, models, and parsers

---

## Contributing

Contributions, improvements, and additional LangChain examples are always welcome.

---

## License

This repository is intended for educational and learning purposes.

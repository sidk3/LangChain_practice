from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs=[
    "Novak Djokovic is an impeccable tennis player.",
    "Roger Federer is a Swiss professional tennis player.",
    "Rafael Nadal is known for his prowess on clay courts."
]

query="Which tennis player is from Switzerland?"

doc_emb=embedding.embed_documents(docs)
query_emb=embedding.embed_query(query)

print(cosine_similarity([query_emb],doc_emb))

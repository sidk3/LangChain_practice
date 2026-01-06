from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("filename.pdf")

docs=loader.load()

print(docs)
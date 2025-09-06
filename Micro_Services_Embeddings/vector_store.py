from typing import List
from langchain.schema import Document
#from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore(texts: List[str], embeddings: Embeddings):
    docs = [Document(page_content=t) for t in texts]
    return FAISS.from_documents(docs, embedding=embeddings)
from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#embedding = HuggingFaceEmbeddings(),
class VectorDB:
    def __init__(self,
        documents = None,
        vector_db: Union[Chroma, FAISS] = Chroma,
        
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
 
        ) -> None:
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db(documents)
    def _build_db(self, documents):
        if not documents or len(documents) == 0:
            raise ValueError("No documents provided for building the vector database.")
        db = self.vector_db.from_documents(documents=documents,
                                        embedding=self.embedding)
        return db
    def get_retriever (self,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 10}
        ):
        retriever = self.db.as_retriever (search_type=search_type,
        search_kwargs=search_kwargs)
        return retriever
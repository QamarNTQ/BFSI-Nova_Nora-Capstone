import os 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT,"data","chroma_chunk_1200_150")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = Chroma(persist_directory=VECTOR_DB_PATH,embedding_function=embedding_model)

def search_policy(query, policy_id=None, k=4):
    if policy_id:
        return vector_db.similarity_search(query, k=k, filter={"policy_id": policy_id})
    else:
        return vector_db.similarity_search(query, k=k)
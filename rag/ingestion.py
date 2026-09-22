from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
POLICY_PATH = os.path.join(PROJECT_ROOT,"data","policies")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT,"data","chroma_index_final")


def load_documents(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError("No PDF files found in data/policies/")
    else:
        return documents

    
def add_policy_metadata(documents):
    for document in documents:
        source = document.metadata.get('source','')
        filename = os.path.basename(source)
        policy_id = filename.replace(".pdf","").replace("Policy","")
        document.metadata['policy_id'] = policy_id
    return documents



def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were generated from the PDFs.")
    else:
        return chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_vector_store(chunks, embedding_model):
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    return vector_db


if __name__ == "__main__":
    documents = load_documents(POLICY_PATH)

    documents = add_policy_metadata(documents)

    chunks = create_chunks(documents)

    build_vector_store(chunks, get_embedding_model())
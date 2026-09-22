import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(PROJECT_ROOT, "data", "policies")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data")

MODELS_TO_TEST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1"
]


def load_documents(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No PDF files found in data/policies/")

    return documents


def add_policy_metadata(documents):
    for document in documents:
        source = document.metadata.get("source", "")
        filename = os.path.basename(source)

        policy_id = filename.replace(".pdf", "").replace("Policy", "")

        document.metadata["policy_id"] = policy_id

    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    return chunks


def create_vector_store(chunks, model_name):
    clean_name = model_name.split("/")[-1]

    db_path = os.path.join(
        VECTOR_DB_PATH,
        f"chroma_index_{clean_name}"
    )

    print(f"\nCreating database: {clean_name}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )

    print(f"Database created: {db_path}")


if __name__ == "__main__":

    print("Loading policy documents...")

    documents = load_documents(POLICY_PATH)

    print(f"Loaded {len(documents)} documents.")

    documents = add_policy_metadata(documents)

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    for model_name in MODELS_TO_TEST:
        create_vector_store(chunks, model_name)

    print("\nAll embedding databases created successfully.")
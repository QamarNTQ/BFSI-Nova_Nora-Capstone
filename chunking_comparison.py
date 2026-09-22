import os
import time
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(PROJECT_ROOT, "data", "policies")
BASE_DB_PATH = os.path.join(PROJECT_ROOT, "data")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_CONFIGS = [
    (500, 50),
    (800, 100),
    (1200, 150),
]

EVALUATION_PAIRS = [
    {"query": "What is the maximum coverage limit for accidental damage?", "expected_phrase": "Rs. 500,000", "policy_id": "P001", "category": "Motor Insurance"},
    {"query": "What is the maximum coverage limit for cargo transit loss?", "expected_phrase": "Rs. 750,000", "policy_id": "P002", "category": "Commercial Fleet"},
    {"query": "What is the maximum payout for total vehicle theft?", "expected_phrase": "Rs. 120,000", "policy_id": "P003", "category": "Two-Wheeler"},
    {"query": "What is the annual inpatient hospitalization coverage limit?", "expected_phrase": "Rs. 700,000", "policy_id": "P004", "category": "Health Insurance"},
    {"query": "What is the maximum lump-sum payout for a critical illness diagnosis?", "expected_phrase": "Rs. 1,500,000", "policy_id": "P005", "category": "Critical Illness"},
    {"query": "What is the maximum coverage for geriatric hospitalization?", "expected_phrase": "Rs. 400,000", "policy_id": "P006", "category": "Senior Citizen Health"},
    {"query": "What is the maximum coverage limit for structural civil damage?", "expected_phrase": "Rs. 4,000,000", "policy_id": "P007", "category": "Property Insurance"},
    {"query": "What is the global aggregate coverage limit for the marine infrastructure policy?", "expected_phrase": "Rs. 50,00,00,000", "policy_id": "P008", "category": "Marine Infrastructure"},
    {"query": "What is the maximum aviation hull aggregate coverage limit?", "expected_phrase": "Rs. 75,00,00,000", "policy_id": "P009", "category": "Aviation"},
    {"query": "What is the maximum facility aggregate coverage limit for the data center?", "expected_phrase": "Rs. 90,00,00,000", "policy_id": "P010", "category": "Data Center"},
    {"query": "What is the maximum aggregate coverage limit for the orbital asset?", "expected_phrase": "Rs. 1,20,00,00,000", "policy_id": "P011", "category": "Satellite"},
    {"query": "What is the maximum facility aggregate coverage for the biotechnology laboratory?", "expected_phrase": "Rs. 85,00,00,000", "policy_id": "P012", "category": "Biotechnology"},
    {"query": "What is the maximum network system aggregate coverage limit?", "expected_phrase": "Rs. 1,10,00,00,000", "policy_id": "P013", "category": "Rail Transit"},
    {"query": "What is the maximum fabrication plant aggregate coverage limit?", "expected_phrase": "Rs. 2,50,00,00,000", "policy_id": "P014", "category": "Semiconductor"},
    {"query": "What is the maximum offshore array aggregate coverage limit?", "expected_phrase": "Rs. 1,80,00,00,000", "policy_id": "P015", "category": "Offshore Wind"},
]


def load_documents():
    loader = DirectoryLoader(POLICY_PATH,glob="*.pdf",loader_cls=PyPDFLoader)

    documents = loader.load()

    if not documents:
        raise ValueError("No policy PDFs found.")

    for document in documents:
        source = document.metadata.get("source", "")
        filename = os.path.basename(source)
        policy_id = filename.replace(".pdf", "").replace("Policy", "")
        document.metadata["policy_id"] = policy_id

    return documents


def create_chunks(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    return splitter.split_documents(documents)


def build_vector_store(chunks, chunk_size, chunk_overlap, embedding_model):
    db_name = f"chroma_chunk_{chunk_size}_{chunk_overlap}"
    db_path = os.path.join(BASE_DB_PATH, db_name)

    if os.path.exists(db_path):
        print(f"Using existing DB: {db_name}")
        return Chroma(persist_directory=db_path, embedding_function=embedding_model)

    print(f"Building DB: {db_name}")

    return Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory=db_path)


def evaluate(vector_db, chunk_size, chunk_overlap):
    results = []

    total_time = 0

    for item in EVALUATION_PAIRS:
        start = time.perf_counter()

        docs = vector_db.similarity_search(item["query"],k=3,filter={"policy_id": item["policy_id"]})

        elapsed = time.perf_counter() - start
        total_time += elapsed

        top1_text = docs[0].page_content if docs else ""
        top3_text = " ".join(doc.page_content for doc in docs)

        top1_hit = item["expected_phrase"] in top1_text
        top3_hit = item["expected_phrase"] in top3_text

        results.append({
            "Chunk Size": chunk_size,
            "Chunk Overlap": chunk_overlap,
            "Policy": item["policy_id"],
            "Query": item["query"],
            "Top-1 Hit": int(top1_hit),
            "Top-3 Hit": int(top3_hit),
            "Search Time (s)": elapsed
        })

    df = pd.DataFrame(results)

    summary = {
        "Chunk Size": chunk_size,
        "Chunk Overlap": chunk_overlap,
        "Avg Search Time (s)": df["Search Time (s)"].mean(),
        "Top-1 Accuracy": df["Top-1 Hit"].mean(),
        "Top-3 Accuracy": df["Top-3 Hit"].mean()
    }

    return df, summary


def main():
    print("Loading policy documents...")
    documents = load_documents()

    print(f"Embedding model: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    all_results = []
    summary_results = []

    for chunk_size, chunk_overlap in CHUNK_CONFIGS:
        print(f"\nTesting chunk size={chunk_size}, overlap={chunk_overlap}")

        chunks = create_chunks(documents,chunk_size,chunk_overlap)

        print(f"Created {len(chunks)} chunks")

        vector_db = build_vector_store(chunks,chunk_size,chunk_overlap,embedding_model)

        raw_df, summary = evaluate( vector_db, chunk_size, chunk_overlap)

        all_results.append(raw_df)
        summary_results.append(summary)

        print(
            f"Top-1: {summary['Top-1 Accuracy']:.2%} | "
            f"Top-3: {summary['Top-3 Accuracy']:.2%} | "
            f"Avg Time: {summary['Avg Search Time (s)']:.4f}s"
        )

    raw_results = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_results)

    raw_results.to_csv("chunking_raw_results.csv",index=False)

    summary_df.to_csv("chunking_benchmark.csv",index=False)

    print("\nFinal Results:")
    print(summary_df.to_string(index=False))

    print("\nSaved:")
    print("chunking_benchmark.csv")
    print("chunking_raw_results.csv")


if __name__ == "__main__":
    main()
import os
import time
import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data")

RESULTS_CSV_PATH = os.path.join(BASE_VECTOR_DB_PATH,"embedding_model_benchmark.csv")

RESULTS_TXT_PATH = os.path.join(BASE_VECTOR_DB_PATH,"embedding_model_benchmark.txt")


MODELS_TO_TEST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1"
]


EVALUATION_PAIRS = [
    {
        "query": "What is the maximum coverage limit for accidental damage?",
        "expected_phrase": "Rs. 500,000",
        "policy_id": "P001",
        "category": "Motor Insurance"
    },
    {
        "query": "What is the maximum coverage limit for cargo transit loss?",
        "expected_phrase": "Rs. 750,000",
        "policy_id": "P002",
        "category": "Commercial Fleet"
    },
    {
        "query": "What is the maximum payout for total vehicle theft?",
        "expected_phrase": "Rs. 120,000",
        "policy_id": "P003",
        "category": "Two-Wheeler"
    },
    {
        "query": "What is the annual inpatient hospitalization coverage limit?",
        "expected_phrase": "Rs. 700,000",
        "policy_id": "P004",
        "category": "Health Insurance"
    },
    {
        "query": "What is the maximum lump-sum payout for a critical illness diagnosis?",
        "expected_phrase": "Rs. 1,500,000",
        "policy_id": "P005",
        "category": "Critical Illness"
    },
    {
        "query": "What is the maximum coverage for geriatric hospitalization?",
        "expected_phrase": "Rs. 400,000",
        "policy_id": "P006",
        "category": "Senior Citizen Health"
    },
    {
        "query": "What is the maximum coverage limit for structural civil damage?",
        "expected_phrase": "Rs. 4,000,000",
        "policy_id": "P007",
        "category": "Property Insurance"
    },
    {
        "query": "What is the global aggregate coverage limit for the marine infrastructure policy?",
        "expected_phrase": "Rs. 50,00,00,000",
        "policy_id": "P008",
        "category": "Marine Infrastructure"
    },
    {
        "query": "What is the maximum aviation hull aggregate coverage limit?",
        "expected_phrase": "Rs. 75,00,00,000",
        "policy_id": "P009",
        "category": "Aviation"
    },
    {
        "query": "What is the maximum facility aggregate coverage limit for the data center?",
        "expected_phrase": "Rs. 90,00,00,000",
        "policy_id": "P010",
        "category": "Data Center"
    },
    {
        "query": "What is the maximum aggregate coverage limit for the orbital asset?",
        "expected_phrase": "Rs. 1,20,00,00,000",
        "policy_id": "P011",
        "category": "Satellite"
    },
    {
        "query": "What is the maximum facility aggregate coverage for the biotechnology laboratory?",
        "expected_phrase": "Rs. 85,00,00,000",
        "policy_id": "P012",
        "category": "Biotechnology"
    },
    {
        "query": "What is the maximum network system aggregate coverage limit?",
        "expected_phrase": "Rs. 1,10,00,00,000",
        "policy_id": "P013",
        "category": "Rail Transit"
    },
    {
        "query": "What is the maximum fabrication plant aggregate coverage limit?",
        "expected_phrase": "Rs. 2,50,00,00,000",
        "policy_id": "P014",
        "category": "Semiconductor"
    },
    {
        "query": "What is the maximum offshore array aggregate coverage limit?",
        "expected_phrase": "Rs. 1,80,00,00,000",
        "policy_id": "P015",
        "category": "Offshore Wind"
    }
]


def get_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


def calculate_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator == 0:
        return 0.0

    return np.dot(v1, v2) / denominator


def run_performance_benchmark():

    os.makedirs(BASE_VECTOR_DB_PATH, exist_ok=True)

    all_results = []

    print("Starting embedding model evaluation...")

    for model_name in MODELS_TO_TEST:

        clean_name = model_name.split("/")[-1]

        specific_db_path = os.path.join(BASE_VECTOR_DB_PATH,f"chroma_index_{clean_name}")

        print(f"\nProcessing: {clean_name}")

        if not os.path.exists(specific_db_path):
            print(f"Database not found: {specific_db_path}")
            continue

        try:
            embed_model = get_embedding_model(model_name)

            vector_db = Chroma(persist_directory=specific_db_path,embedding_function=embed_model)

            for i, test_pair in enumerate(EVALUATION_PAIRS):

                query = test_pair["query"]
                expected = test_pair["expected_phrase"]
                policy_id = test_pair["policy_id"]

                start_time = time.time()

                retrieved_docs = vector_db.similarity_search(query,k=3,filter={"policy_id": policy_id})

                retrieval_latency = time.time() - start_time

                top_match_text = (retrieved_docs[0].page_content if retrieved_docs else "")

                found_in_top_1 = (1.0 if expected.lower() in top_match_text.lower() else 0.0)

                found_in_top_3 = 0.0

                for doc in retrieved_docs:
                    if expected.lower() in doc.page_content.lower():
                        found_in_top_3 = 1.0
                        break

                query_vector = embed_model.embed_query(query)
                expected_vector = embed_model.embed_query(expected)

                semantic_similarity = calculate_cosine_similarity(query_vector,expected_vector)

                all_results.append({
                    "Model": clean_name,
                    "Query_ID": i + 1,
                    "Policy_ID": policy_id,
                    "Category": test_pair["category"],
                    "Latency_Sec": retrieval_latency,
                    "Semantic_Similarity": semantic_similarity,
                    "Hit_Top_1": found_in_top_1,
                    "Hit_Top_3": found_in_top_3
                })

                print(
                    f"  Query {i + 1}: "
                    f"Top-1={found_in_top_1}, "
                    f"Top-3={found_in_top_3}, "
                    f"Time={retrieval_latency:.4f}s"
                )

        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    if not all_results:
        print("No results generated.")
        return

    df_raw = pd.DataFrame(all_results)

    df_summary = df_raw.groupby("Model").agg({
        "Latency_Sec": "mean",
        "Semantic_Similarity": "mean",
        "Hit_Top_1": "mean",
        "Hit_Top_3": "mean"
    }).reset_index()

    df_summary.columns = [
        "Model Name",
        "Avg Search Time (s)",
        "Avg Semantic Match Score",
        "Top-1 Accuracy",
        "Top-3 Accuracy"
    ]

    df_raw.to_csv(
        os.path.join(BASE_VECTOR_DB_PATH, "embedding_model_raw_results.csv"),
        index=False
    )

    df_summary.to_csv(
        RESULTS_CSV_PATH,
        index=False
    )

    with open(RESULTS_TXT_PATH, "w") as f:
        f.write("Embedding Model Benchmark\n\n")
        f.write(df_summary.to_string(index=False))

    print("\nEmbedding Model Evaluation\n")
    print(df_summary.to_string(index=False))

    print(f"\nResults saved to: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    run_performance_benchmark()
import asyncio
import time
import logging
import pandas as pd
from agent.agent import evaluate_claim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("model_comparison.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# MODELS
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b"
]


# TEST CLAIMS
CLAIMS = [
    ("C001", "P001", "accidental damage", 20000),
    ("C002", "P001", "accidental damage", 180000),
    ("C003", "P002", "theft", 50000),
    ("C004", "P003", "fire", 75000),
    ("C005", "P002", "theft", 300000),
]


async def run_comparison():

    results = []

    logger.info("=" * 70)
    logger.info("LLM MODEL COMPARISON STARTED")
    logger.info("Models: %s", MODELS)
    logger.info("Number of claims: %d", len(CLAIMS))
    logger.info("=" * 70)

    for model in MODELS:

        logger.info("Starting tests for model: %s", model)

        model_start_time = time.time()

        for claim in CLAIMS:

            customer_id, policy_id, claim_type, claim_amount = claim

            logger.info(
                "Testing | Model=%s | Customer=%s | Policy=%s | "
                "Claim=%s | Amount=Rs.%s",
                model,
                customer_id,
                policy_id,
                claim_type,
                claim_amount
            )

            start_time = time.time()

            try:

                result = await evaluate_claim(
                    customer_id=customer_id,
                    policy_id=policy_id,
                    claim_type=claim_type,
                    claim_amount=claim_amount,
                    model_id=model
                )

                elapsed = time.time() - start_time

                results.append({
                    "model": model,
                    "customer_id": customer_id,
                    "policy_id": policy_id,
                    "claim_type": claim_type,
                    "claim_amount": claim_amount,
                    "success": True,
                    "response_time_seconds": round(elapsed, 2),
                    "result": result,
                    "error": None
                })

                logger.info(
                    "SUCCESS | Model=%s | Customer=%s | Time=%.2f sec",
                    model,
                    customer_id,
                    elapsed
                )

            except Exception as e:

                elapsed = time.time() - start_time

                results.append({
                    "model": model,
                    "customer_id": customer_id,
                    "policy_id": policy_id,
                    "claim_type": claim_type,
                    "claim_amount": claim_amount,
                    "success": False,
                    "response_time_seconds": round(elapsed, 2),
                    "result": None,
                    "error": str(e)
                })

                logger.error(
                    "FAILED | Model=%s | Customer=%s | Time=%.2f sec | Error=%s",
                    model,
                    customer_id,
                    elapsed,
                    str(e)
                )

        model_elapsed = time.time() - model_start_time

        logger.info(
            "Completed model: %s | Total time=%.2f sec",
            model,
            model_elapsed
        )

    # SAVE RESULTS
    df = pd.DataFrame(results)

    logger.info("=" * 70)
    logger.info("LLM MODEL COMPARISON COMPLETED")
    logger.info("Results saved to: llm_model_comparison.csv")
    logger.info("Logs saved to: model_comparison.log")
    logger.info("=" * 70)

    print("\nFinal Results:")
    print(
        df[
            [
                "model",
                "customer_id",
                "success",
                "response_time_seconds"
            ]
        ].to_string(index=False)
    )

    df.to_csv(
        "llm_model_comparison.csv",
        index=False
    )

if __name__ == "__main__":
    asyncio.run(run_comparison())
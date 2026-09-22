import streamlit as st
import asyncio
from agent.agent import evaluate_claim
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

st.title("Insurance Claims Processing")

customer_id = st.text_input("Customer ID")
policy_id = st.text_input("Policy ID")

claim_type = st.selectbox("Claim Type", ["Accidental damage", "Theft", "Fire", "Natural Calamity"])
claim_amount = st.number_input("Claim Amount", min_value=0.0, step=1000.0)

logger.info(f"Claim evaluation requested: customer={customer_id}, policy={policy_id}")

if st.button("Evaluate Claim"):
    if not customer_id or not policy_id or claim_amount <= 0:
        st.error("Please provide valid claim details.")
    else:
        with st.spinner("Evaluating claim..."):
            result = asyncio.run(evaluate_claim(customer_id, policy_id, claim_type, claim_amount, "qwen/qwen3.8-27b"))

            logger.info(f"Claim evaluation completed: customer={customer_id}, recommendation={result.get('recommendation', 'N/A')}")

        st.subheader("Claim Evaluation")

        # Core Metrics Display
        st.write("**Recommendation:- **", result.get("recommendation", ""))
        st.write("**Reason:- **", result.get("reason", ""))
        
        fraud_risk = result.get("fraud_risk", "")
        
        if "HIGH" in fraud_risk:
            risk_level = "HIGH RISK"
        elif "MEDIUM" in fraud_risk:
            risk_level = "MEDIUM RISK"
        elif "LOW" in fraud_risk:
            risk_level = "LOW RISK"
        else:
            risk_level = "RISK ASSESSMENT INSUFFICIENT"

        st.markdown(f"# {risk_level}")
        st.markdown(f"**Assessment Context:** {fraud_risk}")
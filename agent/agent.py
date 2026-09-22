import os
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_agent
from langgraph.runtime import Runtime
import json


from dotenv import load_dotenv
import asyncio

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_ID = "qwen/qwen3.8-27b"


@before_agent
def validate_claim(state: AgentState, runtime: Runtime):
    messages = state['messages']
    content = messages[-1].content
    required = ["Customer ID:", "Policy ID:", "Claim Type:", "Claim Amount:"]
    if not all(field in content for field in required):
        return {
            "messages": [{"role":"assistant", "content": "Invalid claim: required claim information is missing."}]
        }

    try:
        amount = float(content.split("Claim Amount:")[1].split()[1].replace(",",'').replace("Rs.","").replace(" ",''))
    except (ValueError, IndexError):
        return {
            "messages": [{
                            "role":"assistant",
                            "content":"Invalid claim amount."
                        }]
        }

    if amount <=0:
        return {
           "messages": [{
                           "role":"assistant",
                           "content":"Claim amount must be greater than zero."
                       }]
        }
    return None



async def evaluate_claim(customer_id: str, policy_id: str, claim_type: str, claim_amount: float, model_id: str):
    client = MultiServerMCPClient(
        {
            "claims_server": {
                "url":"http://127.0.0.1:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    print("Available MCP tools:")
    for tool in tools:
        print("-", tool.name)

    print("creating llm")
    llm = ChatGroq(model=model_id, temperature=0, groq_api_key=GROQ_API_KEY)

    print("Creating agent")
    agent = create_agent(model=llm, 
                         tools=tools,
                         middleware=[validate_claim],
                         system_prompt = 
                         """
                        You are an insurance claims processing agent.
                        For every claim evaluation:
                        1. Use policy_coverage to verify whether the claim is covered, 
                           then this info will be passed in fraud_risk tool for checking fraud.
                        2. Use claim_history to retrieve the customer's previous claims.
                        3. Use fraud_risk to assess the fraud risk.

                        Do not invent information.
                        After gathering the required evidence, provide:

                        1. Policy Coverage Assessment
                        2. Claim History Summary
                        3. Fraud Risk Assessment
                        4. Final Recommendation: APPROVE, REJECT, or REFER
                        5. Brief Reason

                        Once you have completed all tool calls and gathered all required data, 
                        your absolute final response to the user MUST contain ONLY a valid JSON object 
                        matching the schema below. Do not include any conversational pleasantries, 
                        markdown text styling outside the json block, or trailing explanations 
                        before or after the JSON structure.
                        {
                            "policy_coverage": "Summary of coverage status...",
                            "claim_history": "Summary of previous history...",
                            "fraud_risk": "Summary of fraud indicators...",
                            "recommendation": "APPROVE" or "REJECT" or "REFER",
                            "reason": "Brief explanation for the recommendation choice..."
                        }
                         """)

    claim = f"""
                Customer ID: {customer_id}
                Policy ID: {policy_id}
                Claim Type: {claim_type}
                Claim Amount: Rs. {claim_amount}
            """

    print("Invoking agent...")
    import time
    start_time = time.time()


    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": claim
                }
            ]  
        }
    )

    print("Agent invocation complete!")
    
    # Safely extract the raw string text 
    raw_response = result['messages'][-1].content
    print("\nFinal Agent Response:\n")
    print(raw_response)

    print("Agent took '",time.time() - start_time, "' seconds to complete agent invocation")

    # Cleaning up markdown wrappers if the LLM adds ```json wrappers anyway
    cleaned_response = raw_response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    try:
        data = json.loads(cleaned_response)
        return data
    except json.JSONDecodeError:
        print("ERROR: Response was not valid JSON. Returning raw string.")
        return {"raw_output": raw_response}


if __name__ == "__main__":
    result = asyncio.run(evaluate_claim("C001", "P001", "Accidental", 20000, "qwen/qwen3.8-27b"))
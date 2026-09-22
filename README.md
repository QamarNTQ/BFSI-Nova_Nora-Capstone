# BFSI Claims Processing Agent

This project is a claims-triage assistant for insurance handlers. It uses policy-document retrieval, three MCP tools, and one LangChain agent to produce a recommendation for human review. The agent never makes the final claims decision.

## Architecture

- `rag/ingestion.py` loads the policy PDFs, adds policy metadata, chunks the text, and builds the Chroma index.
- `rag/retriever.py` searches the persisted Chroma index for a policy-specific coverage clause.
- `mcp_server/server.py` exposes policy coverage, claim history, and fraud-risk tools over streamable HTTP.
- `agent/agent.py` calls the tools and returns a JSON recommendation.
- `app.py` provides the Streamlit claims-handler UI.

## Setup

Use Python 3.10 or later and create a virtual environment from a clean clone:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your-groq-api-key
```

Build the policy index once, or rebuild it after changing files in `data/policies`:

```powershell
python rag/ingestion.py
```

Start the MCP server in one terminal:

```powershell
python -m mcp_server.server
```

Start the UI in another terminal:

```powershell
streamlit run app.py
```

The MCP server must be listening at `http://127.0.0.1:8000/mcp` before a claim is submitted.

## Input and output

The UI accepts a customer ID, policy ID, claim type, and positive claim amount. The agent checks policy evidence, prior claim history, and fraud signals, then returns `APPROVE`, `REJECT`, or `REFER` with a reason. Every recommendation remains subject to human claims-handler review.


## Known limitations

The recommendation is generated from retrieved policy evidence and tool results, but the current policy tool returns evidence for the agent to interpret rather than calculating a deterministic coverage boolean.
# Multi-Agent Research System

A multi-agent workflow built with LangGraph and LangChain. It runs a supervised pipeline (research to analysis to report to evaluation) and exposes it via FastAPI, with a Streamlit UI.

## What it does

Agents:
- Supervisor: routes the workflow
- Researcher: gathers web research and sources
- Analyst: synthesizes findings
- Writer: generates the final report
- Evaluator: scores the report (structured output) and stores feedback

Outputs:
- Task status and results are available via the API.
- Reports are also written to `reports/` as `.txt` and `.json`.

## Requirements

- Python 3.10+ recommended
- API keys:
  - `OPENAI_API_KEY`: LLM provider (recommended for researcher tool-calling)
  - `GROQ_API_KEY`: LLM provider (used by other agents depending on configuration)
  - `GALILEO_API_KEY`: required for evaluation (evaluator will fail without it)
  - `TAVILY_API_KEY`: required for web search

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set the following environment variables:

```bash
export OPENAI_API_KEY="..."
export GROQ_API_KEY="..."
export GALILEO_API_KEY="..."
export TAVILY_API_KEY="..."
```

## Run (FastAPI backend)

```bash
python api.py
```

Then open:
- API docs: `http://localhost:8000/docs`

## Run (Streamlit UI)

Start the API first, then:

```bash
streamlit run streamlit_app.py
```

## Run (CLI)

```bash
python main.py "Your question here"
```

## UI Screenshots

The Streamlit UI provides a clean interface for interacting with the multi-agent system:

### Workflow Progress and Routing Updates
![Workflow Progress](UI%20Result%20Snapshot/result1.png)

### Evaluation Results and Scores
![Evaluation Results](UI%20Result%20Snapshot/result2.png)

### Final Report Display
![Final Report](UI%20Result%20Snapshot/result3.png)



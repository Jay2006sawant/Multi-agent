# Multi-Agent Study Planner

A LangChain + LangGraph based multi-agent system that generates a practical study roadmap from a user's learning goal.

This project is designed for the "Build a Multi-Agent System" assignment and follows the required architecture:
- 4 agents with clear roles
- LangGraph workflow with nodes and edges
- Shared state across agents
- Dynamic terminal input
- Single Python entry file: `multi_agent_system.py`

## Use Case

Given a goal such as:

```text
I want to learn machine learning in 4 weeks with 2 hours daily.
```

the system collaborates across multiple agents to produce:
- learner profile and constraints
- topic breakdown (beginner/intermediate/advanced)
- 4-week study plan
- review and improvement notes

## Tech Stack

- Python 3.10+
- LangChain
- LangGraph
- langchain-ollama
- python-dotenv
- Ollama (optional but recommended for real LLM output)

## Project Structure

```text
Multi-agent/
├── multi_agent_system.py   # Main assignment implementation
├── requirements.txt
├── .env.example
└── README.md
```

## Agent Design

`multi_agent_system.py` contains five nodes in the graph:

1. `profile_agent` - extracts user constraints and learner profile
2. `curriculum_agent` - creates a topic roadmap
3. `planner_agent` - builds a week-by-week actionable plan
4. `reviewer_agent` - critiques and improves the plan
5. `formatter_agent` - combines all outputs into a final report

## Workflow (LangGraph)

```text
profile_agent -> curriculum_agent -> planner_agent -> reviewer_agent -> formatter_agent -> END
```

All intermediate outputs are stored in shared `AgentState` and passed to the next node.

## Setup

```bash
cd /home/jay/Multi-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run with Ollama (Real LLM Output)

Start Ollama server:

```bash
ollama serve
```

In another terminal, pull model:

```bash
ollama pull llama3.2
```

Optional env config:

```bash
cp .env.example .env
```

Then run:

```bash
python3 multi_agent_system.py
```

## Run Without Ollama (Fallback/Mock)

This project includes resilient fallback responses, so it can still run even when Ollama/model is unavailable.

You can explicitly force mock mode:

```bash
USE_MOCK_MODE=true python3 multi_agent_system.py
```

## Example Output Sections

The terminal output contains:
- `User Goal`
- `1) Learner Profile Agent Output`
- `2) Curriculum Agent Output`
- `3) Planner Agent Output`
- `4) Reviewer Agent Output`

## Assignment Checklist Mapping

- 3-4 agents with clear roles: Yes (4 core role agents)
- LangGraph nodes and edges: Yes
- Shared state/context passing: Yes (`AgentState`)
- Main function: Yes (`main()`)
- Dynamic user input: Yes (`input(...)`)
- Single Python file: Yes (`multi_agent_system.py`)

## Troubleshooting

- **Ollama server not running**
  - Run: `ollama serve`
- **Model missing**
  - Run: `ollama pull llama3.2`
- **No internet / model pull fails**
  - Use: `USE_MOCK_MODE=true python3 multi_agent_system.py`
- **Virtual environment not active**
  - Run: `source .venv/bin/activate`

## Notes

- `.venv` and `.env` are ignored in git through `.gitignore`.
- This repository is focused on assignment clarity, code readability, and reproducible terminal demo.

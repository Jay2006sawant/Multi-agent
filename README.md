# Multi-Agent Travel Itinerary Generator

A deterministic multi-agent system built with LangChain + LangGraph that converts a travel request into a practical day-wise itinerary.

## Why This Project

This repository is built for the assignment requirement: build one multi-agent system using LangChain and LangGraph with clear role separation and orchestration.

## Use Case

Input example:

```text
Plan a 5 day trip to Jaipur with budget 25000 and focus on food and culture.
```

Output includes:
- traveler profile and constraints
- destination strategy
- day-wise itinerary
- budget optimization tips

## Tech Stack

- Python 3.10+
- LangChain
- LangGraph

## Project Structure

```text
Multi-agent/
├── multi_agent_system.py
├── requirements.txt
└── README.md
```

## Agent Roles

`multi_agent_system.py` contains these role-based agents:
1. `traveler_profile_agent` - extracts trip duration, budget, planning preferences
2. `destination_research_agent` - creates destination-focused activity plan
3. `itinerary_agent` - builds a day-by-day schedule
4. `budget_optimizer_agent` - adds spend-control and risk-control suggestions
5. `formatter_agent` - prepares final combined response

## LangGraph Workflow

```text
traveler_profile_agent -> destination_research_agent -> itinerary_agent -> budget_optimizer_agent -> formatter_agent -> END
```

Each node reads and updates a shared `AgentState`.

## Setup

```bash
cd /home/jay/Multi-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 multi_agent_system.py
```

## Assignment Requirement Mapping

- 3-4 agents with clear roles: Yes (4 core agents)
- LangGraph nodes and edges: Yes
- Shared state/context passing: Yes (`AgentState`)
- `main()` function: Yes
- Dynamic user input: Yes (`input(...)`)
- Single Python file: Yes (`multi_agent_system.py`)

## Notes

- This implementation is deterministic for reliable demo output.
- `.venv` is ignored by `.gitignore`.

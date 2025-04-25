"""
Multi-agent assignment implementation using LangChain + LangGraph.

Use case: Travel itinerary generator.
"""

from __future__ import annotations

import re
from typing import Dict, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    user_request: str
    traveler_profile: str
    destination_plan: str
    itinerary_plan: str
    budget_tips: str
    final_output: str


def _extract_trip_days(user_request: str) -> int:
    match = re.search(r"(\d+)\s*[- ]?\s*day", user_request, flags=re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return 4


def _extract_budget(user_request: str) -> int:
    match = re.search(r"(?:budget|rs|inr)\s*[:=]?\s*(\d{3,7})", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 30000


def _extract_destination(user_request: str) -> str:
    patterns = [
        r"(?:trip\s+to|travel\s+to|visit)\s+([A-Za-z][A-Za-z ]{1,30}?)(?=\s+(?:with|for|in|on|and|budget)\b|[.,]|$)",
        r"(?:to|in)\s+([A-Za-z][A-Za-z ]{1,30}?)(?=\s+(?:with|for|on|and|budget)\b|[.,]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_request, flags=re.IGNORECASE)
        if not match:
            continue
        destination = match.group(1).strip(" .,!?:;")
        if len(destination) >= 3:
            return destination.title()
    return "Your Chosen City"


def traveler_profile_agent(state: AgentState) -> Dict[str, str]:
    """Agent 1: Extract traveler profile and trip constraints."""
    user_request = state["user_request"]
    days = _extract_trip_days(user_request)
    budget = _extract_budget(user_request)

    profile = (
        f"- Trip duration: {days} days\n"
        f"- Budget estimate: INR {budget}\n"
        "- Travel style: Balanced sightseeing + local experiences\n"
        "- Planning preference: Practical daily schedule with realistic pacing"
    )
    return {"traveler_profile": profile}


def destination_research_agent(state: AgentState) -> Dict[str, str]:
    """Agent 2: Build a destination-focused activity plan."""
    destination = _extract_destination(state["user_request"])
    plan = (
        f"Destination: {destination}\n"
        "- Must include: iconic attraction, local market, local food spot\n"
        "- Add one cultural experience and one relaxed evening activity\n"
        "- Keep commute-friendly grouping of places by area"
    )
    return {"destination_plan": plan}


def itinerary_agent(state: AgentState) -> Dict[str, str]:
    """Agent 3: Convert constraints into day-wise itinerary."""
    days = _extract_trip_days(state["user_request"])
    destination = _extract_destination(state["user_request"])

    sections = []
    for day in range(1, days + 1):
        sections.append(
            f"Day {day}:\n"
            f"- Morning: Key {destination} sightseeing cluster\n"
            "- Afternoon: Local food + nearby exploration\n"
            "- Evening: Relaxed activity and next-day prep"
        )
    return {"itinerary_plan": "\n\n".join(sections)}


def budget_optimizer_agent(state: AgentState) -> Dict[str, str]:
    """Agent 4: Provide budget and risk-control recommendations."""
    budget = _extract_budget(state["user_request"])
    tips_template = PromptTemplate.from_template(
        "Budget limit INR {budget}. Provide concise budget control tips."
    )
    _ = tips_template.format(budget=budget)

    tips = (
        f"- Set daily spend cap around INR {max(1000, budget // max(1, _extract_trip_days(state['user_request'])))}.\n"
        "- Book local transport passes instead of repeated cab hires.\n"
        "- Keep one buffer block for delays or weather changes.\n"
        "- Pre-select two affordable food options near each sightseeing cluster."
    )
    return {"budget_tips": tips}


def formatter_agent(state: AgentState) -> Dict[str, str]:
    """Final node: present all agent outputs as one report."""
    final_report = (
        "=== Multi-Agent Travel Itinerary Generator ===\n\n"
        f"Traveler Request:\n{state['user_request']}\n\n"
        f"1) Traveler Profile Agent Output:\n{state['traveler_profile']}\n\n"
        f"2) Destination Research Agent Output:\n{state['destination_plan']}\n\n"
        f"3) Itinerary Agent Output:\n{state['itinerary_plan']}\n\n"
        f"4) Budget Optimizer Agent Output:\n{state['budget_tips']}\n"
    )
    return {"final_output": final_report}


def build_graph():
    """Construct LangGraph workflow with shared state across agents."""
    workflow = StateGraph(AgentState)

    workflow.add_node("traveler_profile_agent", traveler_profile_agent)
    workflow.add_node("destination_research_agent", destination_research_agent)
    workflow.add_node("itinerary_agent", itinerary_agent)
    workflow.add_node("budget_optimizer_agent", budget_optimizer_agent)
    workflow.add_node("formatter_agent", formatter_agent)

    workflow.set_entry_point("traveler_profile_agent")
    workflow.add_edge("traveler_profile_agent", "destination_research_agent")
    workflow.add_edge("destination_research_agent", "itinerary_agent")
    workflow.add_edge("itinerary_agent", "budget_optimizer_agent")
    workflow.add_edge("budget_optimizer_agent", "formatter_agent")
    workflow.add_edge("formatter_agent", END)

    return workflow.compile()


def main():
    user_input = input("Enter your travel request (dynamic input): ").strip()
    if not user_input:
        raise ValueError("Input cannot be empty.")

    graph = build_graph()

    initial_state: AgentState = {
        "user_request": user_input,
        "traveler_profile": "",
        "destination_plan": "",
        "itinerary_plan": "",
        "budget_tips": "",
        "final_output": "",
    }

    result = graph.invoke(initial_state)
    print("\n" + result["final_output"])


if __name__ == "__main__":
    main()

"""
Multi-agent assignment implementation using LangChain + LangGraph.

Use case: Study planner assistant.
"""

from __future__ import annotations

import os
from typing import Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    user_goal: str
    profile: str
    topic_breakdown: str
    study_plan: str
    review_notes: str
    final_output: str


def _build_llm() -> ChatOpenAI:
    """Create an LLM client with a safe default model."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def profile_agent(state: AgentState) -> Dict[str, str]:
    """Agent 1: Understand learner profile and constraints."""
    llm = _build_llm()
    prompt = (
        "You are a learner-profile agent. Extract constraints and goals from the user request. "
        "Return concise bullet points for: learner level, target, time constraints, and preferences."
    )
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_goal"])])
    return {"profile": response.content}


def curriculum_agent(state: AgentState) -> Dict[str, str]:
    """Agent 2: Convert goals into a topic roadmap."""
    llm = _build_llm()
    prompt = (
        "You are a curriculum designer agent. Using the profile below, create a practical topic breakdown. "
        "Organize into beginner, intermediate, advanced sections with 3-5 items each.\n\n"
        f"Profile:\n{state['profile']}"
    )
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_goal"])])
    return {"topic_breakdown": response.content}


def planner_agent(state: AgentState) -> Dict[str, str]:
    """Agent 3: Generate a week-by-week actionable plan."""
    llm = _build_llm()
    prompt = (
        "You are a study-planner agent. Build a 4-week plan with daily tasks, revision checkpoints, "
        "and one mini-project. Keep it realistic and focused.\n\n"
        f"Profile:\n{state['profile']}\n\n"
        f"Topic Breakdown:\n{state['topic_breakdown']}"
    )
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_goal"])])
    return {"study_plan": response.content}


def reviewer_agent(state: AgentState) -> Dict[str, str]:
    """Agent 4: Improve quality and detect overload/risk."""
    llm = _build_llm()
    prompt = (
        "You are a reviewer agent. Critique the plan for overload, missing fundamentals, and unclear tasks. "
        "Then provide improved final recommendations in concise bullets.\n\n"
        f"Profile:\n{state['profile']}\n\n"
        f"Topic Breakdown:\n{state['topic_breakdown']}\n\n"
        f"Study Plan:\n{state['study_plan']}"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"review_notes": response.content}


def formatter_agent(state: AgentState) -> Dict[str, str]:
    """Final node: present all agent outputs as one report."""
    final_report = (
        "=== Multi-Agent Study Planner ===\n\n"
        f"User Goal:\n{state['user_goal']}\n\n"
        f"1) Learner Profile Agent Output:\n{state['profile']}\n\n"
        f"2) Curriculum Agent Output:\n{state['topic_breakdown']}\n\n"
        f"3) Planner Agent Output:\n{state['study_plan']}\n\n"
        f"4) Reviewer Agent Output:\n{state['review_notes']}\n"
    )
    return {"final_output": final_report}


def build_graph():
    """Construct LangGraph workflow with shared state across agents."""
    workflow = StateGraph(AgentState)

    workflow.add_node("profile_agent", profile_agent)
    workflow.add_node("curriculum_agent", curriculum_agent)
    workflow.add_node("planner_agent", planner_agent)
    workflow.add_node("reviewer_agent", reviewer_agent)
    workflow.add_node("formatter_agent", formatter_agent)

    workflow.set_entry_point("profile_agent")
    workflow.add_edge("profile_agent", "curriculum_agent")
    workflow.add_edge("curriculum_agent", "planner_agent")
    workflow.add_edge("planner_agent", "reviewer_agent")
    workflow.add_edge("reviewer_agent", "formatter_agent")
    workflow.add_edge("formatter_agent", END)

    return workflow.compile()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running this script."
        )

    user_input = input("Enter your learning goal (dynamic input): ").strip()
    if not user_input:
        raise ValueError("Input cannot be empty.")

    graph = build_graph()

    initial_state: AgentState = {
        "user_goal": user_input,
        "profile": "",
        "topic_breakdown": "",
        "study_plan": "",
        "review_notes": "",
        "final_output": "",
    }

    result = graph.invoke(initial_state)
    print("\n" + result["final_output"])


if __name__ == "__main__":
    main()

"""
Multi-agent assignment implementation using LangChain + LangGraph.

Use case: Study planner assistant.
"""

from __future__ import annotations

import os
from typing import Dict, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    user_goal: str
    profile: str
    topic_breakdown: str
    study_plan: str
    review_notes: str
    final_output: str


def _use_mock_mode() -> bool:
    """Enable mock mode explicitly or when no Ollama server is reachable."""
    return os.getenv("USE_MOCK_MODE", "false").lower() == "true"


def _build_llm() -> ChatOllama:
    """Create an Ollama client with a local default model."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, base_url=base_url, temperature=0.3)


def profile_agent(state: AgentState) -> Dict[str, str]:
    """Agent 1: Understand learner profile and constraints."""
    if _use_mock_mode():
        return {
            "profile": (
                "- Learner level: Beginner-to-intermediate\n"
                "- Target: Complete requested learning goal with practical readiness\n"
                "- Time constraints: Follow timeline from user goal\n"
                "- Preference: Hands-on + revision-based schedule"
            )
        }
    try:
        llm = _build_llm()
        prompt = (
            "You are a learner-profile agent. Extract constraints and goals from the user request. "
            "Return concise bullet points for: learner level, target, time constraints, and preferences."
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_goal"])])
        return {"profile": response.content}
    except Exception:
        return {
            "profile": (
                "- Learner level: Not explicitly provided\n"
                "- Target: " + state["user_goal"] + "\n"
                "- Time constraints: Derived from user prompt\n"
                "- Preference: Structured daily plan with checkpoints"
            )
        }


def curriculum_agent(state: AgentState) -> Dict[str, str]:
    """Agent 2: Convert goals into a topic roadmap."""
    if _use_mock_mode():
        return {
            "topic_breakdown": (
                "Beginner:\n"
                "- Fundamentals and core concepts\n"
                "- Problem-solving basics\n"
                "- Tooling and environment setup\n\n"
                "Intermediate:\n"
                "- Pattern-based practice\n"
                "- Timed exercises and analysis\n"
                "- Mini implementation tasks\n\n"
                "Advanced:\n"
                "- Mixed difficulty challenges\n"
                "- Mock interview/project simulation\n"
                "- Weak-area reinforcement"
            )
        }
    try:
        llm = _build_llm()
        prompt = (
            "You are a curriculum designer agent. Using the profile below, create a practical topic breakdown. "
            "Organize into beginner, intermediate, advanced sections with 3-5 items each.\n\n"
            f"Profile:\n{state['profile']}"
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_goal"])])
        return {"topic_breakdown": response.content}
    except Exception:
        return {
            "topic_breakdown": (
                "Beginner:\n- Learn fundamentals\n- Practice easy tasks\n\n"
                "Intermediate:\n- Solve medium tasks\n- Build speed and consistency\n\n"
                "Advanced:\n- Solve hard tasks\n- Simulate real evaluation rounds"
            )
        }


def planner_agent(state: AgentState) -> Dict[str, str]:
    """Agent 3: Generate a week-by-week actionable plan."""
    if _use_mock_mode():
        return {
            "study_plan": (
                "Week 1: Foundation\n"
                "- Daily: 90 min concept study + 30 min practice\n"
                "- Checkpoint: End-of-week quiz\n\n"
                "Week 2: Core practice\n"
                "- Daily: 2 focused practice blocks\n"
                "- Checkpoint: Timed set\n\n"
                "Week 3: Applied work\n"
                "- Daily: Mixed practice + review log\n"
                "- Mini-project/mock task implementation\n\n"
                "Week 4: Revision and assessment\n"
                "- Daily: Weak-topic drills + mocks\n"
                "- Final review and improvement plan"
            )
        }
    try:
        llm = _build_llm()
        prompt = (
            "You are a study-planner agent. Build a 4-week plan with daily tasks, revision checkpoints, "
            "and one mini-project. Keep it realistic and focused.\n\n"
            f"Profile:\n{state['profile']}\n\n"
            f"Topic Breakdown:\n{state['topic_breakdown']}"
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_goal"])])
        return {"study_plan": response.content}
    except Exception:
        return {
            "study_plan": (
                "Week 1: Basics and setup\n"
                "Week 2: Core exercises\n"
                "Week 3: Intermediate practice + mini project\n"
                "Week 4: Revision, mocks, and final improvement"
            )
        }


def reviewer_agent(state: AgentState) -> Dict[str, str]:
    """Agent 4: Improve quality and detect overload/risk."""
    if _use_mock_mode():
        return {
            "review_notes": (
                "- Keep one buffer day weekly to avoid overload.\n"
                "- Track daily progress in a short log.\n"
                "- Spend 20% time on revision.\n"
                "- Increase difficulty only after consistency."
            )
        }
    try:
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
    except Exception:
        return {
            "review_notes": (
                "- Reduce daily load if fatigue appears.\n"
                "- Keep one revision checkpoint every week.\n"
                "- Focus first on weak areas from practice results."
            )
        }


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
    load_dotenv()

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

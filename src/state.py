"""State definition for the multi-agent system."""

from typing import Dict
from langgraph.graph import MessagesState


class SupervisorState(MessagesState):
    """State for the multi-agent system."""
    
    next_agent: str = ""
    routing_reason: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""
    evaluation_scores: Dict[str, float] = {}
    evaluation_feedback: str = ""
    evaluation_passed: bool = False
    sources: list = []  # List of source URLs


"""Routing logic for the multi-agent workflow."""

from typing import Literal
from langgraph.graph import END

from .state import SupervisorState


def create_router() -> callable:
    """Creates and returns the router function."""
    
    def router(state: SupervisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "evaluator", "__end__"]:
        """Routes to next agent based on state."""
        
        next_agent = state.get("next_agent", "supervisor")
        
        # Allow evaluator to run even if task_complete is True (evaluator is read-only)
        if next_agent == "evaluator":
            return "evaluator"
        
        # End if explicitly requested or if task is complete (but evaluator already ran)
        if next_agent == "end" or state.get("task_complete", False):
            return END
            
        if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
            return next_agent
            
        return "supervisor"
    
    return router


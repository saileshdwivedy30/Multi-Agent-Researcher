"""Graph/workflow setup for the multi-agent system."""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import SupervisorState
from .router import create_router
from .agents import supervisor_agent, researcher_agent, analyst_agent, writer_agent, evaluator_agent
from .config import get_judge_llm, get_researcher_llm
from langchain_core.language_models import BaseChatModel


def create_workflow(llm: BaseChatModel, enable_checkpointing: bool = True):
    """Creates and compiles the multi-agent workflow graph with optional checkpointing."""
    
    # Initialize agents with LLM
    supervisor = supervisor_agent(llm)
    
    # Researcher with optional web search - use dedicated LLM with better function calling support
    use_web_search = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
    researcher_llm = get_researcher_llm()  # OpenAI only (tool calling)
    researcher = researcher_agent(researcher_llm, use_web_search=use_web_search)
    
    analyst = analyst_agent(llm)
    writer = writer_agent(llm)
    
    # Initialize evaluator with dedicated judge_llm
    judge_llm = get_judge_llm()
    evaluator = evaluator_agent(judge_llm)
    
    # Create workflow
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("writer", writer)
    workflow.add_node("evaluator", evaluator)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Get router
    router = create_router()
    
    # Add routing for all nodes
    for node in ["supervisor", "researcher", "analyst", "writer", "evaluator"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "supervisor": "supervisor",
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "evaluator": "evaluator",
                END: END
            }
        )
    
    # Add checkpointing if enabled
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


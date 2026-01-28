"""Supervisor agent that manages and coordinates other agents."""

from typing import Dict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from ..state import SupervisorState
from ..utils import invoke_with_retry, track_performance


def create_supervisor_chain(llm: BaseChatModel):
    """Creates the supervisor decision chain."""
    
    supervisor_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a supervisor managing a team of agents:

1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights  
3. Writer - Creates reports and summaries

Based on the current state and conversation, decide which agent should work next.
If the task is complete, respond with 'DONE'.

Current state:
- Has research data: {has_research}
- Has analysis: {has_analysis}
- Has report: {has_report}

Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.
"""
        ),
        ("human", "{task}")
    ])
    
    return supervisor_prompt | llm


def supervisor_agent(llm: BaseChatModel) -> callable:
    """Returns the supervisor agent function configured with the LLM."""
    
    @track_performance
    def agent(state: SupervisorState) -> Dict:
        """Supervisor decides next agent using Groq LLM."""
        
        try:
            messages = state["messages"]
            
            # Get task from current_task state if set, otherwise extract from first HumanMessage
            task = state.get("current_task", "")
            if not task and messages:
                # Find the first HumanMessage to get the original task
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        task = msg.content
                        break
            
            # Fallback if no task found
            if not task:
                task = "No task provided"
            
            # Check what's been completed
            has_research = bool(state.get("research_data", ""))
            has_analysis = bool(state.get("analysis", ""))
            has_report = bool(state.get("final_report", ""))
            
            # Debug: Print state for troubleshooting
            print(f"Supervisor state check - has_research: {has_research}, has_analysis: {has_analysis}, has_report: {has_report}")
            if has_research:
                research_len = len(state.get("research_data", ""))
                print(f"  Research data length: {research_len} characters")
            
            # Get LLM decision (with retry)
            chain = create_supervisor_chain(llm)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    decision = chain.invoke({
                        "task": task,
                        "has_research": has_research,
                        "has_analysis": has_analysis,
                        "has_report": has_report
                    })
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Supervisor chain failed (attempt {attempt + 1}/{max_retries}), retrying: {e}")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Parse decision (used only as fallback if state is unclear)
            decision_text = decision.content.strip().lower()
            
            # Determine next agent - prioritize state checks over LLM decision
            # Only end if we actually have a final report
            if has_report:
                next_agent = "end"
                supervisor_msg = "Supervisor: All tasks complete! Great work team."
                routing_reason = "report complete"
            # If no research, always go to researcher (ignore LLM if it says done)
            elif not has_research:
                next_agent = "researcher"
                supervisor_msg = "Supervisor: Let's start with research. Assigning to Researcher..."
                print(f"Supervisor routing: researcher (no research data found)")
                routing_reason = "no research data found"
            # If research done but no analysis, go to analyst
            elif has_research and not has_analysis:
                next_agent = "analyst"
                supervisor_msg = "Supervisor: Research done. Time for analysis. Assigning to Analyst..."
                print(f"Supervisor routing: analyst (research complete, analysis pending)")
                routing_reason = "research complete, analysis pending"
            # If analysis done but no report, go to writer
            elif has_analysis and not has_report:
                next_agent = "writer"
                supervisor_msg = "Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
                print(f"Supervisor routing: writer (analysis complete, report pending)")
                routing_reason = "analysis complete, report pending"
            # Fallback to LLM decision if state is unclear
            elif "researcher" in decision_text:
                next_agent = "researcher"
                supervisor_msg = "Supervisor: Assigning to Researcher..."
                print(f"Supervisor routing: researcher (LLM fallback)")
                routing_reason = "LLM fallback"
            elif "analyst" in decision_text:
                next_agent = "analyst"
                supervisor_msg = "Supervisor: Assigning to Analyst..."
                print(f"Supervisor routing: analyst (LLM fallback)")
                routing_reason = "LLM fallback"
            elif "writer" in decision_text:
                next_agent = "writer"
                supervisor_msg = "Supervisor: Assigning to Writer..."
                print(f"Supervisor routing: writer (LLM fallback)")
                routing_reason = "LLM fallback"
            else:
                # Only end if LLM explicitly says done AND we have a report
                next_agent = "end"
                supervisor_msg = "Supervisor: Task seems complete."
                print(f"Supervisor routing: end (LLM says done)")
                routing_reason = "LLM says done"
            
            return {
                "messages": [AIMessage(content=supervisor_msg)],
                "next_agent": next_agent,
                "routing_reason": routing_reason,
                "current_task": task
            }
            
        except Exception as e:
            error_msg = f"Supervisor encountered an error: {str(e)}"
            print(f"Error in supervisor: {e}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "next_agent": "supervisor",
                "current_task": state.get("current_task", ""),
                "error": str(e)
            }
    
    return agent


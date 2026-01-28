"""Analyst agent that analyzes data and provides insights."""

from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from ..state import SupervisorState
from ..utils import invoke_with_retry, track_performance


def analyst_agent(llm: BaseChatModel) -> callable:
    """Returns the analyst agent function configured with the LLM."""
    
    @track_performance
    def agent(state: SupervisorState) -> Dict:
        """Analyst uses Groq to analyze the research."""
        
        try:
            research_data = state.get("research_data", "")
            task = state.get("current_task", "")
            
            if not research_data:
                return {
                    "messages": [AIMessage(content="Analyst: No research data available to analyze.")],
                    "analysis": "",
                    "next_agent": "supervisor",
                    "error": "No research data"
                }
            
            # Create analysis prompt
            analysis_prompt = f"""As a data analyst, analyze this research data and provide insights:

Research Data:
{research_data}

Provide:
1. Key insights and patterns
2. Strategic implications
3. Risks and opportunities
4. Recommendations

Focus on actionable insights related to: {task}"""
            
            # Get analysis from LLM (with retry and caching)
            analysis_response = invoke_with_retry(
                llm,
                [HumanMessage(content=analysis_prompt)],
                use_cache=True
            )
            analysis = analysis_response.content
            
            # Create agent message
            agent_message = (
                f"Analyst: I've completed the analysis.\n\n"
                f"Top insights:\n{analysis[:400]}..."
            )
            
            return {
                "messages": [AIMessage(content=agent_message)],
                "analysis": analysis,
                "next_agent": "supervisor"
            }
            
        except Exception as e:
            error_msg = f"Analyst encountered an error: {str(e)}"
            print(f"Error in analyst: {e}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "analysis": "",
                "next_agent": "supervisor",
                "error": str(e)
            }
    
    return agent


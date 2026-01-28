"""Writer agent that creates reports and summaries."""

from typing import Dict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from ..state import SupervisorState
from ..utils import invoke_with_retry, track_performance


def writer_agent(llm: BaseChatModel) -> callable:
    """Returns the writer agent function configured with the LLM."""
    
    @track_performance
    def agent(state: SupervisorState) -> Dict:
        """Writer uses Groq to create final report."""
        
        try:
            research_data = state.get("research_data", "")
            analysis = state.get("analysis", "")
            task = state.get("current_task", "")
            
            if not research_data or not analysis:
                return {
                    "messages": [AIMessage(content="Writer: Missing research data or analysis. Cannot create report.")],
                    "final_report": "",
                    "next_agent": "supervisor",
                    "error": "Missing required data"
                }
            
            # Create writing prompt
            writing_prompt = f"""As a professional writer, create an executive report based on:

Task: {task}

Research Findings:
{research_data[:1000]}

Analysis:
{analysis[:1000]}

Create a well-structured report with:
1. Executive Summary
2. Key Findings  
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise."""
            
            # Get report from LLM (with retry and caching)
            report_response = invoke_with_retry(
                llm,
                [HumanMessage(content=writing_prompt)],
                use_cache=True
            )
            report = report_response.content
            
            # Create final formatted report
            final_report = f"""
FINAL REPORT
{'='*50}
Topic: {task}
{'='*50}

{report}
"""
            
            return {
                "messages": [AIMessage(content="Writer: Report complete! See below for the full document.")],
                "final_report": final_report,
                "next_agent": "evaluator",  # Route to evaluator instead of supervisor
                "task_complete": True
            }
            
        except Exception as e:
            error_msg = f"Writer encountered an error: {str(e)}"
            print(f"Error in writer: {e}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "final_report": "",
                "next_agent": "supervisor",
                "task_complete": False,
                "error": str(e)
            }
    
    return agent


"""Evaluator agent that evaluates the final report using observational evaluation with structured outputs."""

import os
from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ..state import SupervisorState
from ..utils import invoke_with_retry, track_performance

try:
    from galileo import galileo_context
    GALILEO_AVAILABLE = True
except ImportError:
    GALILEO_AVAILABLE = False
    print("Warning: Galileo SDK not installed. Install with: pip install galileo")


class EvaluationScores(BaseModel):
    """Structured output for evaluation scores."""
    factual_consistency: float = Field(ge=0, le=10, description="Score for factual consistency (0-10)")
    completeness: float = Field(ge=0, le=10, description="Score for completeness (0-10)")
    clarity: float = Field(ge=0, le=10, description="Score for clarity (0-10)")
    actionability: float = Field(ge=0, le=10, description="Score for actionability (0-10)")
    feedback: str = Field(description="Overall feedback on the report")


def evaluator_agent(judge_llm: BaseChatModel) -> callable:
    """Returns the evaluator agent function configured with the judge_llm."""
    
    # Use structured output for reliable parsing
    llm_with_structure = judge_llm.with_structured_output(EvaluationScores)
    
    @track_performance
    def agent(state: SupervisorState) -> Dict:
        """Evaluator uses judge_llm to evaluate the final report."""
        
        try:
            final_report = state.get("final_report", "")
            task = state.get("current_task", "")
            
            if not final_report:
                return {
                    "messages": [AIMessage(content="Evaluator: No report found to evaluate.")],
                    "next_agent": "supervisor",
                    "evaluation_scores": {},
                    "evaluation_feedback": "No report available for evaluation.",
                    "evaluation_passed": False,
                }
            
            # Create evaluation prompt
            evaluation_prompt = f"""You are an expert evaluator. Evaluate the following report on the given task.

Task: {task}

Report:
{final_report[:3000]}

Evaluate the report on the following criteria (score each from 0-10):

1. Factual Consistency: Are the facts accurate and consistent?
2. Completeness: Does it address all aspects of the task?
3. Clarity: Is it well-structured and easy to understand?
4. Actionability: Are the recommendations clear and actionable?

Provide scores for each metric and overall feedback."""
            
            if not GALILEO_AVAILABLE:
                raise ImportError(
                    "Galileo SDK is required for evaluation. "
                    "Install it with: pip install galileo"
                )
            
            # Initialize Galileo
            galileo_api_key = os.getenv("GALILEO_API_KEY")
            if not galileo_api_key:
                raise ValueError(
                    "GALILEO_API_KEY environment variable is required. "
                    "Set it in your .env file: GALILEO_API_KEY=your-api-key"
                )
            
            try:
                project_name = os.getenv("GALILEO_PROJECT", "multiagent-evaluation")
                log_stream_name = os.getenv("GALILEO_LOG_STREAM", "evaluator-logs")
                
                galileo_context.init(
                    project=project_name,
                    log_stream=log_stream_name,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Galileo: {e}. "
                    "Please check your GALILEO_API_KEY and network connection."
                ) from e
            
            # Get structured evaluation from judge_llm
            try:
                logger = galileo_context.get_logger_instance()
                
                # Get structured evaluation (with retry and caching)
                evaluation_result: EvaluationScores = invoke_with_retry(
                    llm_with_structure,
                    [HumanMessage(content=evaluation_prompt)],
                    use_cache=False
                )
                
                # Extract scores and feedback from structured output
                scores = {
                    "factual consistency": evaluation_result.factual_consistency,
                    "completeness": evaluation_result.completeness,
                    "clarity": evaluation_result.clarity,
                    "actionability": evaluation_result.actionability,
                }
                
                avg_score = sum(scores.values()) / len(scores) if scores else 0.0
                evaluation_passed = avg_score >= 7.0
                feedback = evaluation_result.feedback
                
                # Log evaluation trace with metadata
                logger.add_single_llm_span_trace(
                    input=evaluation_prompt,
                    output=str(evaluation_result.model_dump()),
                    model="judge_llm",
                    name="evaluator_evaluation",
                    metadata={
                        "agent": "evaluator",
                        "task": task[:200] if len(task) > 200 else task,
                        "evaluation_scores": str(scores),
                        "evaluation_passed": str(evaluation_passed),
                        "average_score": str(avg_score),
                    },
                    tags=["evaluator", "evaluation", "structured_output"]
                )
                
                # Flush to ensure data is sent to Galileo
                galileo_context.flush()
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to log evaluation to Galileo: {e}"
                ) from e
            
            # Create agent message
            status_text = "PASSED" if evaluation_passed else "FAILED"
            agent_message = (
                f"Evaluator: Evaluation complete. "
                f"Average score: {avg_score:.1f}/10. "
                f"{status_text}"
            )
            
            return {
                "messages": [AIMessage(content=agent_message)],
                "next_agent": "supervisor",
                "evaluation_scores": scores,
                "evaluation_feedback": feedback,
                "evaluation_passed": evaluation_passed,
            }
            
        except Exception as e:
            error_msg = f"Evaluator encountered an error: {str(e)}"
            print(f"Error in evaluator: {e}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "next_agent": "supervisor",
                "evaluation_scores": {},
                "evaluation_feedback": f"Error during evaluation: {str(e)}",
                "evaluation_passed": False,
                "error": str(e)
            }
    
    return agent

"""Pydantic models for FastAPI requests and responses."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRequest(BaseModel):
    """Request model for creating a new task."""
    task: str = Field(..., description="The research task/question to process")
    use_web_search: bool = Field(default=True, description="Whether to use web search for research (default: True)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task": "What are the benefits of AI in healthcare?",
                "use_web_search": True
            }
        }


class EvaluationScoresResponse(BaseModel):
    """Evaluation scores response model."""
    factual_consistency: float = Field(..., ge=0, le=10)
    completeness: float = Field(..., ge=0, le=10)
    clarity: float = Field(..., ge=0, le=10)
    actionability: float = Field(..., ge=0, le=10)
    average_score: float = Field(..., ge=0, le=10)
    passed: bool


class TaskProgress(BaseModel):
    """Task progress information."""
    current_agent: Optional[str] = None
    completed_agents: List[str] = Field(default_factory=list)
    total_agents: int = 5
    events: List[str] = Field(
        default_factory=list,
        description="Human-readable progress events (e.g., supervisor routing decisions).",
    )


class TaskResult(BaseModel):
    """Task result model."""
    final_report: str
    evaluation_scores: Optional[Dict[str, float]] = None
    evaluation_feedback: Optional[str] = None
    evaluation_passed: Optional[bool] = None
    conversation_history: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_report": "Executive Report...",
                "evaluation_scores": {
                    "factual consistency": 8.5,
                    "completeness": 8.0,
                    "clarity": 9.0,
                    "actionability": 8.0
                },
                "evaluation_passed": True,
                "conversation_history": ["Message 1", "Message 2"],
                "sources": ["https://example.com"]
            }
        }


class TaskResponse(BaseModel):
    """Response model for task status and results."""
    task_id: str
    status: TaskStatus
    task: str
    created_at: datetime
    updated_at: datetime
    progress: Optional[TaskProgress] = None
    result: Optional[TaskResult] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123",
                "status": "completed",
                "task": "What are the benefits of AI in healthcare?",
                "created_at": "2025-01-28T10:00:00",
                "updated_at": "2025-01-28T10:00:15",
                "progress": {
                    "current_agent": None,
                    "completed_agents": ["supervisor", "researcher", "analyst", "writer", "evaluator"],
                    "total_agents": 5
                },
                "result": {
                    "final_report": "Executive Report...",
                    "evaluation_scores": {
                        "factual consistency": 8.5,
                        "completeness": 8.0,
                        "clarity": 9.0,
                        "actionability": 8.0
                    },
                    "evaluation_passed": True
                }
            }
        }


class TaskCreateResponse(BaseModel):
    """Response model for task creation."""
    task_id: str
    status: TaskStatus
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123",
                "status": "processing",
                "message": "Workflow started successfully"
            }
        }

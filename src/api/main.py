"""FastAPI application for multi-agent system."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .models import (
    TaskRequest,
    TaskResponse,
    TaskCreateResponse,
    TaskStatus,
)
from .task_store import task_store
from .workflow_runner import run_workflow

app = FastAPI(
    title="Multi-Agent AI System API",
    description="API for the multi-agent research and report generation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent AI System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/tasks", response_model=TaskCreateResponse, status_code=202)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    Create a new task and start workflow execution in background.
    
    Returns immediately with task_id. Use GET /api/v1/tasks/{task_id} to check status and get results.
    
    **Note:** use_web_search defaults to True if not specified.
    
    **Workflow:**
    1. POST /api/v1/tasks → Returns task_id immediately
    2. GET /api/v1/tasks/{task_id} → Poll for status and results
    3. Reports are also automatically saved to reports/ directory
    """
    task_id = task_store.create_task(request.task)
    
    use_web_search = getattr(request, 'use_web_search', True)
    
    print(f"\n{'='*60}")
    print(f"New task created: {task_id}")
    print(f"Task: {request.task}")
    print(f"Web search enabled: {use_web_search}")
    print(f"{'='*60}\n")
    
    background_tasks.add_task(
        run_workflow,
        task_id=task_id,
        task=request.task,
        use_web_search=use_web_search
    )
    
    return TaskCreateResponse(
        task_id=task_id,
        status=TaskStatus.PROCESSING,
        message="Workflow started successfully. Use GET /api/v1/tasks/{task_id} to check status."
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Get task status and result.
    
    - **task_id**: Unique identifier for the task
    """
    task = task_store.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return TaskResponse(**task)


@app.get("/api/v1/tasks", response_model=List[TaskResponse])
async def list_tasks():
    """List all tasks."""
    tasks = task_store.get_all_tasks()
    return [TaskResponse(**task) for task in tasks.values()]


@app.delete("/api/v1/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task.
    
    - **task_id**: Unique identifier for the task
    """
    if not task_store.delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return {"message": f"Task {task_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

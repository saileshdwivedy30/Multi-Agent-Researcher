"""In-memory task store for tracking workflow execution."""

import uuid
from typing import Dict, Optional
from datetime import datetime
from enum import Enum

from .models import TaskStatus, TaskProgress, TaskResult


class TaskStore:
    """In-memory store for task tracking."""
    
    def __init__(self):
        self._tasks: Dict[str, dict] = {}
    
    def create_task(self, task: str) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "task_id": task_id,
            "task": task,
            "status": TaskStatus.PENDING,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "progress": TaskProgress(total_agents=5),
            "result": None,
            "error": None,
        }
        return task_id
    
    def update_status(self, task_id: str, status: TaskStatus, error: Optional[str] = None):
        """Update task status."""
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = status
            self._tasks[task_id]["updated_at"] = datetime.now()
            if error:
                self._tasks[task_id]["error"] = error
    
    def update_progress(self, task_id: str, progress: TaskProgress):
        """Update task progress."""
        if task_id in self._tasks:
            existing_progress = self._tasks[task_id].get("progress")
            existing_events = []
            if existing_progress is not None and getattr(existing_progress, "events", None) is not None:
                existing_events = list(existing_progress.events)

            data = progress.model_dump() if hasattr(progress, "model_dump") else dict(progress)  # type: ignore[arg-type]
            data["events"] = existing_events
            self._tasks[task_id]["progress"] = TaskProgress(**data)
            self._tasks[task_id]["updated_at"] = datetime.now()

    def append_event(self, task_id: str, event: str):
        """Append a human-readable progress event to the task."""
        if task_id not in self._tasks:
            return
        progress = self._tasks[task_id].get("progress") or TaskProgress(total_agents=5)
        events = list(getattr(progress, "events", []) or [])
        events.append(event)

        data = progress.model_dump() if hasattr(progress, "model_dump") else dict(progress)  # type: ignore[arg-type]
        data["events"] = events
        self._tasks[task_id]["progress"] = TaskProgress(**data)
        self._tasks[task_id]["updated_at"] = datetime.now()
    
    def set_result(self, task_id: str, result: TaskResult):
        """Set task result."""
        if task_id in self._tasks:
            self._tasks[task_id]["result"] = result
            self._tasks[task_id]["status"] = TaskStatus.COMPLETED
            self._tasks[task_id]["updated_at"] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, dict]:
        """Get all tasks."""
        return self._tasks.copy()
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False


# Global task store instance
task_store = TaskStore()

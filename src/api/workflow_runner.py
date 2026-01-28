"""Workflow runner for executing tasks in background."""

import hashlib
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from ..config import get_llm
from ..graph import create_workflow
from .task_store import task_store
from .models import TaskStatus, TaskProgress, TaskResult


def extract_progress_from_state(state: Dict[str, Any]) -> TaskProgress:
    """Extract progress information from workflow state."""
    completed_agents = []
    
    # Check which agents have completed
    if state.get("research_data"):
        completed_agents.append("researcher")
    if state.get("analysis"):
        completed_agents.append("analyst")
    if state.get("final_report"):
        completed_agents.append("writer")
    if state.get("evaluation_scores"):
        completed_agents.append("evaluator")
    
    # Determine current agent based on next_agent
    next_agent = state.get("next_agent", "supervisor")
    current_agent = None
    if next_agent != "end" and next_agent not in completed_agents:
        current_agent = next_agent
    
    return TaskProgress(
        current_agent=current_agent,
        completed_agents=completed_agents,
        total_agents=5,
    )

def _merge_state(current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort merge of LangGraph node updates into our running state snapshot."""
    for key, value in (update or {}).items():
        if key == "messages" and isinstance(value, list):
            current.setdefault("messages", [])
            current["messages"].extend(value)
            continue
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            # Shallow merge for dict-like fields
            current[key].update(value)  # type: ignore[union-attr]
            continue
        current[key] = value
    return current


def extract_sources_from_messages(messages: list) -> list:
    """Extract source URLs from messages."""
    sources = []
    for msg in messages:
        content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        # Look for URLs in content
        if "http" in content.lower():
            # Simple extraction - could be improved
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            sources.extend(urls)
    return list(set(sources))  # Remove duplicates


def run_workflow(task_id: str, task: str, use_web_search: bool = True):
    """Run the workflow for a given task."""
    try:
        print(f"\n{'='*60}")
        print(f"Starting workflow for task: {task_id}")
        print(f"Task: {task}")
        print(f"Web search: {use_web_search}")
        print(f"{'='*60}\n")
        
        # Update status to processing
        task_store.update_status(task_id, TaskStatus.PROCESSING)
        task_store.append_event(task_id, "Workflow started")
        
        # Set environment variable for web search
        import os
        os.environ["USE_WEB_SEARCH"] = "true" if use_web_search else "false"
        
        # Initialize LLM and workflow
        print("Initializing LLM and workflow...")
        llm = get_llm()
        graph = create_workflow(llm, enable_checkpointing=True)
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_task": task
        }
        
        # Generate thread_id for checkpointing
        thread_id = hashlib.md5(f"{task_id}_{task}".encode()).hexdigest()[:8]
        config = {"configurable": {"thread_id": thread_id}}
        
        print("Executing workflow...\n")
        sys.stdout.flush()
        
        # Stream workflow updates so the UI can show live progress.
        running_state: Dict[str, Any] = dict(initial_state)
        last_progress = extract_progress_from_state(running_state)
        task_store.update_progress(task_id, last_progress)

        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            # `event` is typically: {"node_name": {state_updates...}}
            if not isinstance(event, dict):
                continue
            for node_name, node_update in event.items():
                if not isinstance(node_update, dict):
                    continue

                running_state = _merge_state(running_state, node_update)

                # Update progress snapshot
                progress = extract_progress_from_state(running_state)
                task_store.update_progress(task_id, progress)

                # Log supervisor routing decisions as progress events
                if node_name == "supervisor":
                    to_agent = node_update.get("next_agent") or running_state.get("next_agent")
                    reason = node_update.get("routing_reason") or running_state.get("routing_reason")
                    if to_agent:
                        suffix = f" ({reason})" if reason else ""
                        task_store.append_event(task_id, f"Supervisor → {to_agent}{suffix}")

        # Final state after streaming completes
        response = running_state
        
        print(f"\n{'='*60}")
        print("Workflow completed successfully!")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Extract conversation history
        conversation_history = []
        for msg in response.get("messages", []):
            if hasattr(msg, 'content'):
                conversation_history.append(msg.content)
        
        # Extract sources from state (preferred) or messages (fallback)
        sources = response.get("sources", [])
        if not sources:
            # Fallback: try to extract from messages
            sources = extract_sources_from_messages(response.get("messages", []))
        
        # Print summary
        print(f"Results:")
        print(f"  - Report length: {len(response.get('final_report', ''))} characters")
        print(f"  - Evaluation passed: {response.get('evaluation_passed', False)}")
        print(f"  - Sources found: {len(sources)}")
        if sources:
            print(f"  - Source URLs: {sources[:3]}...")  # Show first 3
        sys.stdout.flush()
        
        # Create result
        result = TaskResult(
            final_report=response.get("final_report", ""),
            evaluation_scores=response.get("evaluation_scores"),
            evaluation_feedback=response.get("evaluation_feedback"),
            evaluation_passed=response.get("evaluation_passed"),
            conversation_history=conversation_history,
            sources=sources
        )
        
        # Update task with result
        task_store.set_result(task_id, result)
        
        # Save report to file
        try:
            reports_dir = Path(__file__).parent.parent.parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Create filename from task (sanitized) and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_slug = task[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
            task_slug = "".join(c for c in task_slug if c.isalnum() or c in "_-")
            filename_base = f"{timestamp}_{task_slug}"
            
            # Save as text file
            text_file = reports_dir / f"{filename_base}.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("MULTI-AGENT AI SYSTEM REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Task: {task}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "=" * 70 + "\n")
                f.write("FINAL REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(result.final_report)
                
                if sources:
                    f.write("\n\n" + "=" * 70 + "\n")
                    f.write("SOURCES\n")
                    f.write("=" * 70 + "\n\n")
                    for i, source in enumerate(sources, 1):
                        f.write(f"{i}. {source}\n")
                
                if result.evaluation_scores:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write("EVALUATION\n")
                    f.write("=" * 70 + "\n\n")
                    for metric, score in result.evaluation_scores.items():
                        f.write(f"  {metric.capitalize()}: {score:.1f}/10\n")
                    avg = sum(result.evaluation_scores.values()) / len(result.evaluation_scores)
                    f.write(f"\n  Average: {avg:.1f}/10\n")
                    f.write(f"  Status: {'PASSED' if result.evaluation_passed else 'FAILED'}\n")
                    if result.evaluation_feedback:
                        f.write(f"\n  Feedback: {result.evaluation_feedback}\n")
            
            # Save as JSON file
            json_file = reports_dir / f"{filename_base}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({
                    "task_id": task_id,
                    "task": task,
                    "generated_at": datetime.now().isoformat(),
                    "final_report": result.final_report,
                    "sources": sources,
                    "evaluation_scores": result.evaluation_scores,
                    "evaluation_feedback": result.evaluation_feedback,
                    "evaluation_passed": result.evaluation_passed,
                    "conversation_history": result.conversation_history,
                }, f, indent=2, ensure_ascii=False)
            
            print(f"Report saved:")
            print(f"   Text: {text_file}")
            print(f"   JSON: {json_file}")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"Warning: Could not save report to file: {e}")
            sys.stdout.flush()
        
        print(f"Task {task_id} marked as completed.\n")
        sys.stdout.flush()
        
    except Exception as e:
        # Update status to failed
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"ERROR in workflow execution for task {task_id}:")
        print(f"{'='*60}")
        print(f"Error: {error_msg}\n")
        task_store.update_status(task_id, TaskStatus.FAILED, error=error_msg)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
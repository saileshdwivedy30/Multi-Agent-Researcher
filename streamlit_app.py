"""Streamlit UI for the multi-agent research system."""

import streamlit as st
import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime

API_BASE_URL = "http://localhost:8000"
API_TASKS_URL = f"{API_BASE_URL}/api/v1/tasks"

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-status {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-running {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .agent-completed {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .agent-pending {
        background-color: #fafafa;
        border-left: 4px solid #9e9e9e;
    }
    .evaluation-score {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .source-link {
        color: #1976d2;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def create_task(task: str) -> Optional[str]:
    """Create a new task via API."""
    try:
        response = requests.post(
            API_TASKS_URL,
            json={"task": task, "use_web_search": True},
            timeout=5
        )
        if response.status_code == 202:
            return response.json()["task_id"]
        else:
            st.error(f"Failed to create task: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please make sure the FastAPI server is running on http://localhost:8000")
        st.info("Start the API server with: `python api.py`")
        return None
    except Exception as e:
        st.error(f"Error creating task: {str(e)}")
        return None


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status from API."""
    try:
        response = requests.get(f"{API_TASKS_URL}/{task_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"Task {task_id} not found")
            return None
        else:
            st.error(f"Error getting task status: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching task status: {str(e)}")
        return None


def get_agent_icon(agent_name: str) -> str:
    """Get icon for agent."""
    return ""


def get_agent_display_name(agent_name: str) -> str:
    """Get display name for agent."""
    names = {
        "supervisor": "Supervisor",
        "researcher": "Researcher",
        "analyst": "Analyst",
        "writer": "Writer",
        "evaluator": "Evaluator"
    }
    return names.get(agent_name, agent_name.title())


def display_progress(progress: Dict[str, Any], current_agent: Optional[str]):
    """Display workflow progress."""
    if not progress:
        return
    
    completed_agents = progress.get("completed_agents", [])
    total_agents = progress.get("total_agents", 5)
    
    # Calculate progress percentage
    progress_pct = len(completed_agents) / total_agents if total_agents > 0 else 0
    
    # Progress bar
    st.progress(progress_pct)
    st.caption(f"Progress: {len(completed_agents)}/{total_agents} agents completed ({progress_pct*100:.0f}%)")
    
    # Agent status
    all_agents = ["supervisor", "researcher", "analyst", "writer", "evaluator"]
    
    for agent in all_agents:
        icon = get_agent_icon(agent)
        name = get_agent_display_name(agent)
        
        if agent in completed_agents:
            status_class = "agent-completed"
            status_text = "Completed"
        elif agent == current_agent:
            status_class = "agent-running"
            status_text = "Running..."
        else:
            status_class = "agent-pending"
            status_text = "Pending"
        
        st.markdown(f"""
            <div class="agent-status {status_class}">
                <strong>{name}</strong>: {status_text}
            </div>
        """, unsafe_allow_html=True)

def display_routing_updates(progress: Dict[str, Any]):
    """Display live supervisor routing updates."""
    if not progress:
        return
    events = progress.get("events") or []
    # Only show supervisor routing lines, in chronological order
    events = [e for e in events if isinstance(e, str) and e.startswith("Supervisor →")]
    if not events:
        return
    st.subheader("Workflow Updates")
    for evt in events[-20:]:
        st.markdown(f"- {evt}")


def display_evaluation_scores(scores: Dict[str, float], passed: bool, feedback: Optional[str]):
    """Display evaluation scores."""
    if not scores:
        return
    
    st.subheader("Evaluation Results")
    
    # Calculate average
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    
    # Score metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Factual Consistency", f"{scores.get('factual consistency', 0):.1f}/10")
    with col2:
        st.metric("Completeness", f"{scores.get('completeness', 0):.1f}/10")
    with col3:
        st.metric("Clarity", f"{scores.get('clarity', 0):.1f}/10")
    with col4:
        st.metric("Actionability", f"{scores.get('actionability', 0):.1f}/10")
    with col5:
        st.metric("Average Score", f"{avg_score:.1f}/10")
    
    # Status badge
    if passed:
        st.success(f"Evaluation PASSED (Average: {avg_score:.1f}/10)")
    else:
        st.warning(f"Evaluation FAILED (Average: {avg_score:.1f}/10)")
    
    # Feedback
    if feedback:
        with st.expander("Evaluation Feedback"):
            st.write(feedback)


def display_sources(sources: list):
    """Display source URLs."""
    if not sources:
        return
    
    st.subheader(f"Sources ({len(sources)})")
    
    with st.expander("View all sources"):
        for i, source in enumerate(sources, 1):
            st.markdown(f"{i}. [{source}]({source})")


def display_report(report: str):
    """Display final report."""
    st.subheader("Final Report")
    
    # Render markdown
    st.markdown(report)
    
    # Download button
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.txt"
    st.download_button(
        label="Download Report",
        data=report,
        file_name=filename,
        mime="text/plain"
    )


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">Multi-Agent Research System</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("API server is not running. Please start it with: `python api.py`")
        st.stop()
    
    # Initialize session state
    if "task_id" not in st.session_state:
        st.session_state.task_id = None
    if "polling" not in st.session_state:
        st.session_state.polling = False
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This system uses multiple AI agents to:
        1. Research your question
        2. Analyze findings
        3. Generate a report
        4. Evaluate quality
        """)
    
    # Main input section
    st.header("Ask a Question")
    
    # Text input
    question = st.text_input(
        "Enter your research question:",
        placeholder="e.g., What are the benefits of AI in healthcare?",
        key="question_input"
    )
    
    # Submit button
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)
    
    # Handle submission
    if submit_button and question:
        with st.spinner("Creating task..."):
            task_id = create_task(question)
            if task_id:
                st.session_state.task_id = task_id
                st.session_state.polling = True
                st.session_state.question = question
                st.rerun()
    
    # Polling and progress display
    if st.session_state.polling and st.session_state.task_id:
        task_id = st.session_state.task_id
        
        # Get task status
        task_data = get_task_status(task_id)
        
        if task_data:
            status = task_data.get("status")
            progress = task_data.get("progress")
            result = task_data.get("result")
            error = task_data.get("error")
            
            # Show progress section
            st.header("Workflow Progress")
            
            if status == "processing":
                display_routing_updates(progress)
                
                # Auto-refresh
                time.sleep(1.5)
                st.rerun()
            
            elif status == "completed":
                st.session_state.polling = False
                
                # Show completion message
                st.success("Workflow completed successfully!")
                
                # Display results
                if result:
                    # Evaluation scores
                    if result.get("evaluation_scores"):
                        display_evaluation_scores(
                            result["evaluation_scores"],
                            result.get("evaluation_passed", False),
                            result.get("evaluation_feedback")
                        )
                    
                    st.divider()
                    
                    # Final report
                    if result.get("final_report"):
                        display_report(result["final_report"])
                    
                    st.divider()
                    
                    # Sources
                    if result.get("sources"):
                        display_sources(result["sources"])
                
                # Reset button
                if st.button("Ask Another Question"):
                    st.session_state.task_id = None
                    st.session_state.polling = False
                    st.rerun()
            
            elif status == "failed":
                st.session_state.polling = False
                st.error("Workflow failed")
                if error:
                    st.error(f"Error: {error}")
                
                if st.button("Try Again"):
                    st.session_state.task_id = None
                    st.session_state.polling = False
                    st.rerun()


if __name__ == "__main__":
    main()

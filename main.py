"""Main entry point for the multi-agent system."""

import sys
from langchain_core.messages import HumanMessage

from src.config import get_llm
from src.graph import create_workflow


def main():
    """Run the multi-agent system."""
    
    # Get task from command line or use default
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "What are the benefits and risks of AI in healthcare?"
    
    print(f"\n{'='*60}")
    print(f"Multi-Agent AI System")
    print(f"{'='*60}")
    print(f"Task: {task}\n")
    
    try:
        # Initialize LLM
        print("Initializing LLM...")
        llm = get_llm()
        
        # Create workflow
        print("Creating workflow graph...")
        graph = create_workflow(llm)
        
        # Run the workflow
        # Initialize state explicitly with both message and current_task
        # to ensure the task is available to all agents
        print("Executing workflow...\n")
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_task": task
        }
        
        # Provide config for checkpointing (required when checkpointing is enabled)
        # Use a simple thread_id based on task hash for consistency
        import hashlib
        thread_id = hashlib.md5(task.encode()).hexdigest()[:8]
        config = {"configurable": {"thread_id": thread_id}}
        
        response = graph.invoke(initial_state, config=config)
        
        # Display results
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)
        
        # Print conversation history
        print("\n--- Conversation History ---")
        for msg in response.get("messages", []):
            print(f"\n{msg.content}")
        
        # Print final report if available
        if response.get("final_report"):
            print("\n" + "="*60)
            print(response["final_report"])
        else:
            print("\nFinal report not generated.")
        
        # Print evaluation results if available
        evaluation_scores = response.get("evaluation_scores", {})
        if evaluation_scores:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            
            print("\n--- Evaluation Scores (0-10) ---")
            for metric, score in evaluation_scores.items():
                print(f"  {metric.capitalize()}: {score:.1f}/10")
            
            avg_score = sum(evaluation_scores.values()) / len(evaluation_scores) if evaluation_scores else 0.0
            print(f"\n  Average Score: {avg_score:.1f}/10")
            
            evaluation_passed = response.get("evaluation_passed", False)
            status = "PASSED" if evaluation_passed else "FAILED"
            print(f"  Status: {status}")
            
            evaluation_feedback = response.get("evaluation_feedback", "")
            if evaluation_feedback:
                print("\n--- Evaluation Feedback ---")
                print(evaluation_feedback)
            
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


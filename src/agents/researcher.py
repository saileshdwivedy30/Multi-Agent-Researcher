"""Researcher agent that gathers information and data with web search capabilities."""

import os
from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from ..state import SupervisorState
from ..utils import invoke_with_retry, track_performance

try:
    from langchain_community.tools import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    raise ImportError("TavilySearchResults not available. Install with: pip install langchain-community")


def researcher_agent(llm: BaseChatModel, use_web_search: bool = True) -> callable:
    """Returns the researcher agent function configured with the LLM and Tavily web search."""
    
    if not use_web_search:
        raise ValueError("Web search is required for the researcher agent. Set use_web_search=True.")
    
    if not TAVILY_AVAILABLE:
        raise ImportError("TavilySearchResults not available. Install with: pip install langchain-community")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. Please set it in a .env file or as an environment variable.\n"
            "Web search is required for the researcher agent."
        )
    
    tools = [TavilySearchResults(max_results=5, api_key=tavily_api_key)]
    llm_with_tools = llm.bind_tools(tools)
    
    @track_performance
    def agent(state: SupervisorState) -> Dict:
        """Researcher uses LLM with tool calling to gather information via web search."""
        
        try:
            task = state.get("current_task", "")
            if not task:
                messages = state.get("messages", [])
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        task = msg.content
                        break
            if not task:
                task = "research topic"
            
            research_prompt = f"""As a research specialist, search the web and provide comprehensive information about: {task}

Use the search tool to find current, accurate information. Include:
1. Key facts and background
2. Current trends or developments (use recent sources)
3. Important statistics or data points
4. Notable examples or case studies
5. Source citations when possible

Be concise but thorough. Cite your sources."""
            
            research_data = ""
            sources = []
            messages = [HumanMessage(content=research_prompt)]
            
            # Step 1: LLM generates structured tool call
            response = invoke_with_retry(llm_with_tools, messages, use_cache=False)
            messages.append(response)
            
            # Step 2: Extract tool calls from response
            tool_calls_to_execute = response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else []
            
            # Step 3: Framework executes tools
            if tool_calls_to_execute:
                print(f"LLM generated {len(tool_calls_to_execute)} tool call(s)")
                for tool_call in tool_calls_to_execute:
                    tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                    tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                    tool_id = tool_call.get("id", "") if isinstance(tool_call, dict) else getattr(tool_call, "id", "")
                    
                    # Find and execute the tool
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                                
                                # Extract sources if available
                                if isinstance(tool_result, list):
                                    for result in tool_result:
                                        if isinstance(result, dict):
                                            if "url" in result:
                                                sources.append(result.get("url", ""))
                                            if "content" in result:
                                                research_data += f"\n\n{result.get('content', '')}"
                                            if "snippet" in result:
                                                research_data += f"\n\n{result.get('snippet', '')}"
                                elif isinstance(tool_result, str):
                                    research_data += f"\n\n{tool_result}"
                                
                                # Step 4: Framework feeds results back to LLM
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_id
                                ))
                                break
                            except Exception as e:
                                print(f"Warning: Tool {tool_name} execution failed: {e}")
                                messages.append(ToolMessage(
                                    content=f"Error: {str(e)}",
                                    tool_call_id=tool_id
                                ))
                                break
                
                # Step 5: LLM processes results
                if messages:
                    final_response = invoke_with_retry(llm_with_tools, messages, use_cache=False)
                    research_data = final_response.content
            else:
                # No tool calls, use regular response
                research_data = response.content
            
            # Add sources to research data if available
            if sources:
                research_data += f"\n\nSources:\n" + "\n".join(f"- {url}" for url in sources[:5])
            
            # Create agent message
            agent_message = f"Researcher: I've completed the research on '{task}'."
            if sources:
                agent_message += f" Found {len(sources)} source(s)."
            agent_message += f"\n\nKey findings:\n{research_data[:500]}..."
            
            print(f"Researcher completed: research_data length = {len(research_data)} characters, sources = {len(sources)}")
            
            return {
                "messages": [AIMessage(content=agent_message)],
                "research_data": research_data,
                "sources": sources,
                "next_agent": "supervisor"
            }
            
        except Exception as e:
            error_msg = f"Researcher encountered an error: {str(e)}"
            print(f"Error in researcher: {e}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "research_data": "",
                "next_agent": "supervisor",
                "error": str(e)
            }
    
    return agent

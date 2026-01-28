"""Researcher agent that gathers information and data with web search capabilities."""

import os
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from ..state import SupervisorState
from ..utils import invoke_with_retry, track_performance

# Try to import web search tools
WEB_SEARCH_AVAILABLE = False
TAVILY_AVAILABLE = False
DUCKDUCKGO_AVAILABLE = False

try:
    from langchain_community.tools import TavilySearchResults
    TAVILY_AVAILABLE = True
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    # Test if DuckDuckGo can actually be initialized (requires ddgs package)
    try:
        _ = DuckDuckGoSearchRun()
        DUCKDUCKGO_AVAILABLE = True
        WEB_SEARCH_AVAILABLE = True
    except (ImportError, Exception):
        pass
except ImportError:
    pass

if not WEB_SEARCH_AVAILABLE:
    print("Warning: Web search tools not available. Install with: pip install langchain-community ddgs")


def researcher_agent(llm: BaseChatModel, use_web_search: bool = True) -> callable:
    """Returns the researcher agent function configured with the LLM and optional web search."""
    
    # Set up tools if web search is enabled
    tools = []
    llm_with_tools = llm
    
    if use_web_search and WEB_SEARCH_AVAILABLE:
        # Try Tavily first (better quality), fallback to DuckDuckGo
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key and TAVILY_AVAILABLE:
            try:
                tools = [TavilySearchResults(max_results=5, api_key=tavily_api_key)]
                llm_with_tools = llm.bind_tools(tools)
            except Exception as e:
                print(f"Warning: Tavily setup failed: {e}. Trying DuckDuckGo...")
                if DUCKDUCKGO_AVAILABLE:
                    try:
                        tools = [DuckDuckGoSearchRun()]
                        llm_with_tools = llm.bind_tools(tools)
                    except Exception as e2:
                        print(f"Warning: DuckDuckGo setup also failed: {e2}. Continuing without web search.")
                        tools = []
                        llm_with_tools = llm
                else:
                    print("Warning: DuckDuckGo not available. Continuing without web search.")
                    tools = []
                    llm_with_tools = llm
        elif DUCKDUCKGO_AVAILABLE:
            # Use free DuckDuckGo if no Tavily API key
            try:
                tools = [DuckDuckGoSearchRun()]
                llm_with_tools = llm.bind_tools(tools)
            except Exception as e:
                print(f"Warning: DuckDuckGo setup failed: {e}. Continuing without web search.")
                tools = []
                llm_with_tools = llm
        else:
            print("Warning: No web search tools available. Continuing without web search.")
            tools = []
            llm_with_tools = llm
    
    @track_performance
    def agent(state: SupervisorState) -> Dict:
        """Researcher uses Groq to gather information, optionally with web search."""
        
        try:
            # Get task from current_task state, or extract from first HumanMessage
            task = state.get("current_task", "")
            if not task:
                messages = state.get("messages", [])
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        task = msg.content
                        break
            if not task:
                task = "research topic"
            
            # Create research prompt
            if tools:
                research_prompt = f"""As a research specialist, search the web and provide comprehensive information about: {task}

Use the search tool to find current, accurate information. Include:
1. Key facts and background
2. Current trends or developments (use recent sources)
3. Important statistics or data points
4. Notable examples or case studies
5. Source citations when possible

Be concise but thorough. Cite your sources."""
            else:
                research_prompt = f"""As a research specialist, provide comprehensive information about: {task}
Include:
1. Key facts and background
2. Current trends or developments
3. Important statistics or data points
4. Notable examples or case studies

Be concise but thorough."""
            
            # Standard tool calling: LLM generates structured tool calls, framework executes
            research_data = ""
            sources = []
            messages = [HumanMessage(content=research_prompt)]
            
            if tools:
                # Standard tool calling: LLM generates structured tool calls, framework executes
                try:
                    # Step 1: LLM generates structured tool call
                    response = invoke_with_retry(llm_with_tools, messages, use_cache=False)
                    messages.append(response)
                    
                    # Step 2: Check if LLM wants to call tools
                    # Extract tool calls from response (handle different formats)
                    tool_calls_to_execute = []
                    
                    # Try different ways to get tool calls
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        tool_calls_to_execute = response.tool_calls
                    elif hasattr(response, 'additional_kwargs'):
                        if 'tool_calls' in response.additional_kwargs:
                            tool_calls_to_execute = response.additional_kwargs['tool_calls']
                        elif 'function_call' in response.additional_kwargs:
                            # Handle function_call format (older API)
                            func_call = response.additional_kwargs['function_call']
                            tool_calls_to_execute = [{
                                "name": func_call.get("name", ""),
                                "args": func_call.get("arguments", {}),
                                "id": f"call_{hash(str(func_call))}"
                            }]
                    
                    # Also check response_metadata
                    if not tool_calls_to_execute and hasattr(response, 'response_metadata'):
                        metadata = response.response_metadata
                        if 'tool_calls' in metadata:
                            tool_calls_to_execute = metadata['tool_calls']
                    
                    # Step 3: Framework executes tools
                    if tool_calls_to_execute:
                        print(f"LLM generated {len(tool_calls_to_execute)} tool call(s)")
                        for tool_call in tool_calls_to_execute:
                            # Extract tool call info
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name", "")
                                tool_args = tool_call.get("args", {})
                                tool_id = tool_call.get("id", "")
                            else:
                                # Handle different tool call formats
                                tool_name = getattr(tool_call, "name", "")
                                tool_args = getattr(tool_call, "args", {})
                                tool_id = getattr(tool_call, "id", "")
                            
                            # Find and execute the tool
                            tool_executed = False
                            for tool in tools:
                                if tool.name == tool_name:
                                    try:
                                        # Framework executes tool
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
                                        tool_executed = True
                                        break
                                    except Exception as e:
                                        print(f"Warning: Tool {tool_name} execution failed: {e}")
                                        messages.append(ToolMessage(
                                            content=f"Error: {str(e)}",
                                            tool_call_id=tool_id
                                        ))
                                        tool_executed = True
                                        break
                            
                            if not tool_executed:
                                print(f"Warning: Tool {tool_name} not found")
                                messages.append(ToolMessage(
                                    content=f"Tool {tool_name} not available",
                                    tool_call_id=tool_id if 'tool_id' in locals() else ""
                                ))
                        
                        # Step 5: LLM processes results
                        if messages:
                            final_response = invoke_with_retry(llm_with_tools, messages, use_cache=False)
                            research_data = final_response.content
                    else:
                        # No tool calls, use regular response
                        research_data = response.content
                        
                except Exception as e:
                    # If tool calling fails, check if it's a Groq-specific issue
                    error_str = str(e).lower()
                    if "tool_use_failed" in error_str or ("function" in error_str and "groq" in error_str):
                        # Groq function calling issue - try with lower temperature or different approach
                        print(f"Warning: Function calling error detected. This may be a Groq limitation.")
                        print("Consider using OPENAI_API_KEY for tool calling support.")
                        # Fall back to regular LLM
                        research_response = invoke_with_retry(
                            llm,
                            [HumanMessage(content=research_prompt)],
                            use_cache=False
                        )
                        research_data = research_response.content
                    else:
                        # Other error - re-raise
                        raise
            else:
                # No tools available, use regular LLM
                research_response = invoke_with_retry(
                    llm,
                    [HumanMessage(content=research_prompt)],
                    use_cache=False
                )
                research_data = research_response.content
            
            # Add sources to research data if available
            if sources:
                research_data += f"\n\nSources:\n" + "\n".join(f"- {url}" for url in sources[:5])
            
            # Create agent message
            agent_message = (
                f"Researcher: I've completed the research on '{task}'."
            )
            if sources:
                agent_message += f" Found {len(sources)} source(s)."
            agent_message += f"\n\nKey findings:\n{research_data[:500]}..."
            
            # Debug: Print research data status
            print(f"Researcher completed: research_data length = {len(research_data)} characters, sources = {len(sources)}")
            
            return {
                "messages": [AIMessage(content=agent_message)],
                "research_data": research_data,
                "sources": sources,  # Store sources in state
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

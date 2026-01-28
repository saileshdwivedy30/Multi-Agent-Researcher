"""Utility functions for the multi-agent system."""

import logging
import time
from functools import wraps
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Configure logging
logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def invoke_with_retry(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    use_cache: bool = True,
) -> Any:
    """Invoke LLM with automatic retry on failure."""
    try:
        return llm.invoke(messages)
    except Exception as e:
        # Don't log verbose errors for Groq function calling issues (handled gracefully)
        error_str = str(e).lower()
        if "tool_use_failed" in error_str or ("function" in error_str and "groq" in error_str.lower()):
            # This will be handled by the caller, don't spam logs
            pass
        else:
            logger.error(f"LLM call failed: {e}", exc_info=True)
        raise


def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(
                f"{func.__name__} completed in {duration:.2f}s",
                extra={"duration": duration, "function": func.__name__},
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {duration:.2f}s: {e}",
                exc_info=True,
            )
            raise
    return wrapper

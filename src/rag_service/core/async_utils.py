"""Async utilities for running CPU-bound operations.

Provides helpers for running CPU-bound operations (like embedding generation)
in a ThreadPoolExecutor to avoid blocking the async event loop.

Example usage:
    from rag_service.core.async_utils import run_in_executor

    # In an async endpoint
    async def query(text: str):
        embedding = await run_in_executor(
            embedding_service.embed_query, text
        )
        return embedding
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")

# Global executor for CPU-bound operations
# Using a module-level variable with lazy initialization
_executor: ThreadPoolExecutor | None = None

# Default number of workers - matches typical CPU count for embedding models
DEFAULT_MAX_WORKERS = 4


def get_executor(max_workers: int = DEFAULT_MAX_WORKERS) -> ThreadPoolExecutor:
    """Get or create the global ThreadPoolExecutor.

    Uses lazy initialization to create the executor on first use.
    The executor is reused across all calls.

    Args:
        max_workers: Maximum number of worker threads.

    Returns:
        The global ThreadPoolExecutor instance.
    """
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Shutdown the global ThreadPoolExecutor.

    Should be called during application shutdown to cleanly
    terminate worker threads.

    Args:
        wait: If True, wait for pending tasks to complete.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None


async def run_in_executor(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a CPU-bound function in a thread pool.

    Executes the given function in a ThreadPoolExecutor to avoid
    blocking the async event loop. This is ideal for CPU-intensive
    operations like embedding generation.

    Args:
        func: The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.

    Example:
        # Instead of blocking:
        embedding = embedding_service.embed_query(text)

        # Use async:
        embedding = await run_in_executor(
            embedding_service.embed_query, text
        )
    """
    loop = asyncio.get_event_loop()
    executor = get_executor()

    # Use partial to handle keyword arguments
    if kwargs:
        func = partial(func, **kwargs)

    return await loop.run_in_executor(executor, func, *args)


async def run_batch_in_executor(
    func: Callable[[list[T]], list[Any]],
    items: list[T],
    batch_size: int = 32,
) -> list[Any]:
    """Run a batch processing function in a thread pool.

    Splits items into batches and processes each batch in the
    executor. Useful for batch embedding operations.

    Args:
        func: Function that processes a list of items.
        items: List of items to process.
        batch_size: Number of items per batch.

    Returns:
        Combined results from all batches.
    """
    results: list[Any] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_result = await run_in_executor(func, batch)
        results.extend(batch_result)

    return results

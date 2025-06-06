"""
Parallel processing utilities for AgenticRAG.
Provides functions for running tasks in parallel.
"""

import concurrent.futures
import logging
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Tuple
from error_handler import handle_error, try_except_decorator, ErrorHandler

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

def get_optimal_thread_count(task_count: int = 1) -> int:
    """
    Determine the optimal number of threads based on system resources and task count.

    Args:
        task_count: Number of tasks to be executed

    Returns:
        Optimal number of worker threads
    """
    import os
    import psutil

    try:
        # Get CPU count
        cpu_count = os.cpu_count() or 4

        # Get available memory
        available_memory = psutil.virtual_memory().available

        # Calculate memory-based thread limit (assuming each thread might use ~50MB)
        memory_thread_limit = max(1, int(available_memory / (50 * 1024 * 1024)))

        # Calculate CPU-based thread limit (use 75% of CPUs to avoid system overload)
        cpu_thread_limit = max(1, int(cpu_count * 0.75))

        # Use the minimum of memory and CPU limits
        system_limit = min(memory_thread_limit, cpu_thread_limit)

        # Also consider the number of tasks (no need for more threads than tasks)
        optimal_count = min(system_limit, task_count)

        logger.debug(f"Optimal thread count: {optimal_count} (CPU limit: {cpu_thread_limit}, Memory limit: {memory_thread_limit}, Tasks: {task_count})")
        return optimal_count

    except Exception as e:
        # If anything goes wrong, fall back to a reasonable default
        logger.warning(f"Error determining optimal thread count: {str(e)}, using default")
        return min(4, task_count)

def run_in_parallel(
    tasks: List[Tuple[Callable[..., R], Tuple, Dict[str, Any]]],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    retry_count: int = 0,
    retry_delay: float = 1.0
) -> List[Optional[R]]:
    """
    Run multiple tasks in parallel using a thread pool.

    Args:
        tasks: List of tuples containing (function, args, kwargs)
        max_workers: Maximum number of worker threads (None = auto)
        timeout: Maximum time to wait for all tasks to complete (None = no timeout)
        retry_count: Number of times to retry failed tasks
        retry_delay: Delay between retries in seconds

    Returns:
        List of results in the same order as the tasks
    """
    results = [None] * len(tasks)

    if not tasks:
        return results

    # Determine optimal thread count if not specified
    if max_workers is None:
        max_workers = get_optimal_thread_count(len(tasks))

    # Track task execution time for performance monitoring
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, (func, args, kwargs) in enumerate(tasks):
            future = executor.submit(func, *args, **kwargs)
            future_to_index[future] = i

        # Wait for tasks to complete
        try:
            for future in concurrent.futures.as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Task {index} raised an exception: {str(e)}")
                    results[index] = None

                    # Retry the task if retry_count > 0
                    if retry_count > 0:
                        retry_task(executor, tasks[index], index, results, retry_count, retry_delay)

        except concurrent.futures.TimeoutError:
            logger.warning(f"Timeout occurred after {timeout} seconds")

    # Log performance metrics
    elapsed_time = time.time() - start_time
    successful_tasks = sum(1 for r in results if r is not None)
    logger.info(f"Parallel execution completed: {successful_tasks}/{len(tasks)} tasks successful in {elapsed_time:.2f}s using {max_workers} workers")

    return results

def retry_task(
    executor: concurrent.futures.ThreadPoolExecutor,
    task: Tuple[Callable[..., R], Tuple, Dict[str, Any]],
    index: int,
    results: List[Optional[R]],
    retry_count: int,
    retry_delay: float
) -> None:
    """
    Retry a failed task.

    Args:
        executor: The thread pool executor
        task: The task to retry (function, args, kwargs)
        index: The index of the task in the results list
        results: The list of results
        retry_count: Number of retries remaining
        retry_delay: Delay between retries in seconds
    """
    func, args, kwargs = task

    for attempt in range(retry_count):
        try:
            # Wait before retrying
            time.sleep(retry_delay)

            # Submit the task again
            logger.info(f"Retrying task {index}, attempt {attempt + 1}/{retry_count}")
            result = func(*args, **kwargs)

            # Update the results
            results[index] = result
            logger.info(f"Retry successful for task {index}")

            # Break out of the retry loop on success
            break

        except Exception as e:
            logger.error(f"Retry {attempt + 1}/{retry_count} for task {index} failed: {str(e)}")

            # Last attempt failed, keep the result as None
            if attempt == retry_count - 1:
                logger.error(f"All retries failed for task {index}")
                results[index] = None

def parallel_document_retrieval(
    query: str,
    vectordbs: List[Any],
    k: int = 8,
    timeout: Optional[float] = 10.0,
    retry_count: int = 1
) -> List[Tuple[Any, int]]:
    """
    Retrieve documents from multiple vector databases in parallel.

    Args:
        query: The query to search for
        vectordbs: List of vector databases to search
        k: Number of documents to retrieve from each database
        timeout: Maximum time to wait for all retrievals to complete
        retry_count: Number of times to retry failed retrievals

    Returns:
        List of tuples containing (document, database_index)
    """
    if not vectordbs:
        return []

    # Create tasks for each vector database
    tasks = []
    for db in vectordbs:
        tasks.append((db.similarity_search, (query, k), {}))

    # Run tasks in parallel with retry capability
    results = run_in_parallel(
        tasks,
        timeout=timeout,
        retry_count=retry_count,
        retry_delay=0.5  # Short delay for retries
    )

    # Create a list of (document, database_index) tuples
    all_docs_with_source = []
    for db_index, docs in enumerate(results):
        if docs:
            # Add each document with its database index
            for doc in docs:
                # Add a source indicator to the metadata
                if 'source_db' not in doc.metadata:
                    doc.metadata['source_db'] = 'temp' if db_index > 0 else 'permanent'
                # Add database index to metadata for better tracking
                doc.metadata['db_index'] = db_index
                all_docs_with_source.append((doc, db_index))

    logger.info(f"Retrieved {len(all_docs_with_source)} documents from {len(vectordbs)} databases")
    return all_docs_with_source

def prioritize_documents(
    docs_with_source: List[Tuple[Any, int]],
    temp_db_boost: float = 2.0
) -> List[Any]:
    """
    Prioritize documents from temporary databases over permanent ones.

    Args:
        docs_with_source: List of tuples containing (document, database_index)
        temp_db_boost: Boost factor for temporary database results (db_index > 0)

    Returns:
        Prioritized list of documents
    """
    if not docs_with_source:
        return []

    # First, separate documents by database type
    temp_docs = [doc for doc, db_index in docs_with_source if db_index > 0]
    perm_docs = [doc for doc, db_index in docs_with_source if db_index == 0]

    # Log the distribution
    logger.info(f"Retrieved {len(temp_docs)} documents from temporary DBs and {len(perm_docs)} from permanent DB")

    # If we have temporary database results, prioritize them
    if temp_docs:
        # Get the total number of documents we want to return
        total_docs = len(temp_docs) + len(perm_docs)
        max_docs = min(total_docs, 20)  # Increased from 16 to 20 for more comprehensive context

        # For very high boost values (4.0+), use only temporary docs if available
        if temp_db_boost >= 4.0 and len(temp_docs) > 0:
            temp_count = min(len(temp_docs), max_docs)
            perm_count = 0
            logger.info(f"High boost factor ({temp_db_boost}): Using only temporary database results")
        # For boost values between 3.0 and 4.0, heavily prioritize temp docs but include some permanent
        elif temp_db_boost >= 3.0 and len(temp_docs) > 0:
            # Use at least 80% temp docs
            temp_proportion = 0.8
            temp_count = min(len(temp_docs), max(int(max_docs * temp_proportion), 1))
            perm_count = min(len(perm_docs), max_docs - temp_count)
            logger.info(f"High boost factor ({temp_db_boost}): Heavily prioritizing temporary database results (80/20 split)")
        else:
            # Calculate the ratio based on the boost factor
            # The higher the boost, the more we favor temporary docs
            # At boost=1.0: equal ratio (50/50)
            # At boost=2.0: 2:1 ratio (67/33)
            # At boost=2.5: 2.5:1 ratio (71/29)

            # Calculate effective weights
            temp_weight = temp_db_boost
            perm_weight = 1.0

            # Calculate proportions
            total_weight = temp_weight + perm_weight
            temp_proportion = temp_weight / total_weight

            # Calculate counts based on proportions
            temp_count = min(len(temp_docs), int(max_docs * temp_proportion))
            # Ensure we have at least some temp docs if available
            temp_count = max(2, temp_count) if len(temp_docs) > 1 else max(1, temp_count) if len(temp_docs) > 0 else 0
            # Remaining slots go to permanent docs
            perm_count = min(len(perm_docs), max_docs - temp_count)

        # Create the final prioritized list
        # Interleave the documents to ensure better context mixing
        prioritized_docs = []

        # If we have a very high boost, just use temp docs first
        if temp_db_boost >= 3.5:
            prioritized_docs.extend(temp_docs[:temp_count])
            if perm_count > 0:
                prioritized_docs.extend(perm_docs[:perm_count])
        else:
            # Interleave documents with preference to temp docs based on boost factor
            temp_idx = 0
            perm_idx = 0

            # Calculate how many temp docs to include per permanent doc
            temp_per_perm = max(1, int(temp_db_boost))

            while len(prioritized_docs) < (temp_count + perm_count):
                # Add temp docs based on the ratio
                for _ in range(temp_per_perm):
                    if temp_idx < temp_count:
                        prioritized_docs.append(temp_docs[temp_idx])
                        temp_idx += 1

                # Add a permanent doc
                if perm_idx < perm_count:
                    prioritized_docs.append(perm_docs[perm_idx])
                    perm_idx += 1

                # If we've used all permanent docs but still have temp docs, add the rest
                if perm_idx >= perm_count and temp_idx < temp_count:
                    prioritized_docs.extend(temp_docs[temp_idx:temp_count])
                    break

                # If we've used all temp docs but still have permanent docs, add the rest
                if temp_idx >= temp_count and perm_idx < perm_count:
                    prioritized_docs.extend(perm_docs[perm_idx:perm_count])
                    break

        logger.info(f"Prioritized {temp_count} temporary DB docs and {perm_count} permanent DB docs with boost factor {temp_db_boost}")
        return prioritized_docs
    else:
        # If no temporary database results, just return the permanent database results
        return perm_docs

def parallel_web_search(
    query: str,
    search_functions: List[Tuple[Callable, Dict[str, Any]]],
    timeout: Optional[float] = 15.0
) -> List[Dict[str, Any]]:
    """
    Perform web searches using multiple search functions in parallel.

    Args:
        query: The search query
        search_functions: List of tuples containing (search_function, kwargs)
        timeout: Maximum time to wait for all searches to complete

    Returns:
        Combined list of search results
    """
    if not search_functions:
        return []

    # Create tasks for each search function
    tasks = []
    for func, kwargs in search_functions:
        tasks.append((func, (query,), kwargs))

    # Run tasks in parallel
    results = run_in_parallel(tasks, timeout=timeout)

    # Combine the results
    all_results = []
    for result in results:
        if result:
            all_results.extend(result)

    return all_results

def parallel_document_processing(
    processor_func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    retry_count: int = 1,
    batch_size: Optional[int] = None
) -> List[Any]:
    """
    Process multiple documents in parallel with improved performance and error handling.

    Args:
        processor_func: Function to process each item
        items: List of items to process
        max_workers: Maximum number of worker threads
        timeout: Maximum time to wait for all processing to complete
        retry_count: Number of times to retry failed processing
        batch_size: Process items in batches of this size (None = process all at once)

    Returns:
        List of processed results
    """
    if not items:
        return []

    # Process in batches if specified
    if batch_size and batch_size > 0 and len(items) > batch_size:
        logger.info(f"Processing {len(items)} items in batches of {batch_size}")
        all_results = []

        # Process each batch
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size} ({len(batch)} items)")

            # Process the batch
            batch_results = parallel_document_processing(
                processor_func,
                batch,
                max_workers=max_workers,
                timeout=timeout,
                retry_count=retry_count,
                batch_size=None  # Prevent recursive batching
            )

            # Add the batch results to the overall results
            all_results.extend(batch_results)

        return all_results

    # Create tasks for each item
    tasks = []
    for item in items:
        tasks.append((processor_func, (item,), {}))

    # Run tasks in parallel with retry capability
    results = run_in_parallel(
        tasks,
        max_workers=max_workers,
        timeout=timeout,
        retry_count=retry_count
    )

    # Flatten the results
    all_results = []
    for result in results:
        if result:
            if isinstance(result, list):
                all_results.extend(result)
            else:
                all_results.append(result)

    return all_results

"""
Search utilities for web search functionality.
Supports multiple search providers including DuckDuckGo and SerpAPI.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from cache_manager import cached_web_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get search configuration from environment
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serpapi").lower()  # Changed default from duckduckgo to serpapi
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TIMELIMIT = os.getenv("WEB_SEARCH_TIMELIMIT", "y")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

@cached_web_search
def search_web(query: str, domain_context: str = "cyber security") -> List[Dict[str, Any]]:
    """
    Search the web using the configured search provider.
    Results are cached to improve performance for repeated queries.

    Args:
        query: The search query
        domain_context: Optional domain context to add to the query

    Returns:
        List of search results as dictionaries with title, link, snippet, and date
    """
    search_query = f"{domain_context} {query}" if domain_context else query

    # Log which search provider is being used
    logger.info(f"Using search provider: {SEARCH_PROVIDER}")

    # Check if SerpAPI key is set when using SerpAPI
    if SEARCH_PROVIDER == "serpapi" and not SERPAPI_KEY:
        logger.warning("SerpAPI selected but no API key provided. Check your .env file or web UI settings.")

    results = []
    if SEARCH_PROVIDER == "serpapi":
        results = search_with_serpapi(search_query)
    else:
        results = search_with_duckduckgo(search_query)

    # Validate results
    logger.info(f"Search returned {len(results)} results")

    # If no results were returned, create a fallback result
    if not results:
        logger.warning("No search results returned, creating fallback result")
        results = [{
            "title": "No search results found",
            "link": "",
            "snippet": "The search did not return any results. Try modifying your query or using different keywords.",
            "date": ""
        }]

    # Ensure all results are dictionaries with the required fields
    validated_results = []
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            logger.warning(f"Result {i} is not a dictionary: {type(result)}")
            # Convert to dictionary
            if isinstance(result, str):
                result = {"title": "Unknown", "link": "#", "snippet": result, "date": ""}
            else:
                result = {"title": "Unknown", "link": "#", "snippet": str(result), "date": ""}

        # Ensure all required fields exist
        validated_result = {
            "title": result.get("title", ""),
            "link": result.get("link", "") or result.get("href", ""),  # Support both link and href
            "snippet": result.get("snippet", "") or result.get("body", ""),  # Support both snippet and body
            "date": result.get("date", "") or result.get("published", "")  # Support both date and published
        }
        validated_results.append(validated_result)

    return validated_results

def search_with_duckduckgo(query: str) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query

    Returns:
        List of search results as dictionaries with title, link, snippet, and date
    """
    try:
        from duckduckgo_search import DDGS

        # Add retry logic for web search
        max_retries = 3
        retry_delay = 2  # seconds
        web_results = []

        for attempt in range(max_retries):
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(
                        query,
                        region='wt-wt',
                        safesearch='on',
                        timelimit=WEB_SEARCH_TIMELIMIT,
                        max_results=WEB_SEARCH_MAX_RESULTS
                    ):
                        if r and isinstance(r, dict):
                            web_results.append({
                                "title": r.get("title", ""),
                                "link": r.get("href", ""),
                                "snippet": r.get("body", ""),
                                "date": r.get("published", "")
                            })

                # If we got results, break the retry loop
                if web_results:
                    break
                else:
                    # If no results were found, create a fallback result
                    logger.warning("No results found in DuckDuckGo response, creating fallback result")
                    web_results.append({
                        "title": "No search results found",
                        "link": "",
                        "snippet": "The search did not return any results. Try modifying your query or using different keywords.",
                        "date": ""
                    })

            except Exception as e:
                logger.warning(f"DuckDuckGo search attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All DuckDuckGo search attempts failed")
                    raise

        return web_results

    except ImportError:
        logger.error("DuckDuckGo search package not installed. Install with: pip install duckduckgo_search")
        return []
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return []

def search_with_serpapi(query: str) -> List[Dict[str, Any]]:
    """
    Search the web using SerpAPI (Google Search).

    Args:
        query: The search query

    Returns:
        List of search results as dictionaries with title, link, snippet, and date
    """
    try:
        # The package is named google-search-results but the import is serpapi
        from serpapi import GoogleSearch

        if not SERPAPI_KEY:
            logger.error("SerpAPI key not found. Please set SERPAPI_KEY in your .env file.")
            return []

        # Add retry logic for SerpAPI
        max_retries = 3
        retry_delay = 2  # seconds
        web_results = []

        for attempt in range(max_retries):
            try:
                # Configure the search
                search_params = {
                    "q": query,
                    "api_key": SERPAPI_KEY,
                    "num": WEB_SEARCH_MAX_RESULTS,
                    "tbm": "nws" if "news" in query.lower() else None,  # Use news search if query contains "news"
                }

                # Remove None values
                search_params = {k: v for k, v in search_params.items() if v is not None}

                # Execute the search
                search = GoogleSearch(search_params)
                results = search.get_dict()

                # Log the response structure for debugging
                logger.info(f"SerpAPI response keys: {results.keys() if isinstance(results, dict) else 'Not a dictionary'}")

                # Process organic results
                if isinstance(results, dict) and "organic_results" in results:
                    logger.info(f"Found {len(results['organic_results'])} organic results")
                    for result in results["organic_results"][:WEB_SEARCH_MAX_RESULTS]:
                        if isinstance(result, dict):
                            web_results.append({
                                "title": result.get("title", ""),
                                "link": result.get("link", ""),
                                "snippet": result.get("snippet", ""),
                                "date": result.get("date", "")
                            })
                        else:
                            logger.warning(f"Organic result is not a dictionary: {type(result)}")

                # Process news results if present
                if isinstance(results, dict) and "news_results" in results:
                    news_data = results["news_results"]
                    if isinstance(news_data, dict) and "results" in news_data:
                        logger.info(f"Found {len(news_data['results'])} news results")
                        for result in news_data["results"][:WEB_SEARCH_MAX_RESULTS]:
                            if isinstance(result, dict):
                                web_results.append({
                                    "title": result.get("title", ""),
                                    "link": result.get("link", ""),
                                    "snippet": result.get("snippet", ""),
                                    "date": result.get("date", "")
                                })
                            else:
                                logger.warning(f"News result is not a dictionary: {type(result)}")
                    else:
                        logger.warning("news_results does not contain a 'results' key or is not a dictionary")

                # If we got results, break the retry loop
                if web_results:
                    break
                else:
                    # If no results were found in the standard fields, check for other result types
                    logger.warning("No standard results found, checking for alternative result types")

                    # Check for knowledge graph results
                    if isinstance(results, dict) and "knowledge_graph" in results:
                        kg = results["knowledge_graph"]
                        if isinstance(kg, dict):
                            # Create a result from knowledge graph data
                            web_results.append({
                                "title": kg.get("title", ""),
                                "link": kg.get("website", "") or "",
                                "snippet": kg.get("description", ""),
                                "date": ""
                            })
                            logger.info("Added knowledge graph result")

                    # Check for answer box results
                    if isinstance(results, dict) and "answer_box" in results:
                        ab = results["answer_box"]
                        if isinstance(ab, dict):
                            # Create a result from answer box data
                            web_results.append({
                                "title": ab.get("title", "") or "Featured Snippet",
                                "link": ab.get("link", "") or "",
                                "snippet": ab.get("snippet", "") or ab.get("answer", ""),
                                "date": ""
                            })
                            logger.info("Added answer box result")

                    # If still no results, create a fallback result
                    if not web_results:
                        logger.warning("No results found in SerpAPI response, creating fallback result")
                        web_results.append({
                            "title": "No search results found",
                            "link": "",
                            "snippet": "The search did not return any results. Try modifying your query or using different keywords.",
                            "date": ""
                        })

            except Exception as e:
                logger.warning(f"SerpAPI search attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All SerpAPI search attempts failed")
                    raise

        return web_results

    except ImportError:
        logger.error("SerpAPI package not installed. Install with: pip install google-search-results")
        logger.info("After installing, you'll need to restart the application.")
        return []
    except Exception as e:
        logger.error(f"Error in SerpAPI search: {str(e)}")
        return []

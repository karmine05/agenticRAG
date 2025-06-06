"""
Unified RAG engine for CTI Agent.
Provides a single, configurable RAG implementation that combines the functionality
of all previous RAG approaches.
"""

import logging
import os
import streamlit as st
from typing import Dict, Any, List, Optional, Union, Tuple
from langchain.schema import Document

from config import config
from model_manager import call_reasoning_model, call_tool_model
from search_utils import search_web
from response_cleaner import clean_response
from cache_manager import cached_document_retrieval
from parallel_utils import parallel_document_retrieval, prioritize_documents
from prompt_templates import create_reasoning_prompt, create_tool_prompt, create_direct_prompt
from vector_store import vector_store_manager
from web_to_temp_db import web_results_to_temp_db

# Set up logging
logger = logging.getLogger(__name__)

@cached_document_retrieval
def retrieve_documents(query: str, vectordb, k: int = 8) -> List[Document]:
    """
    Retrieve documents from the vector database with caching.

    Args:
        query: The query to search for
        vectordb: The vector database to search
        k: The number of documents to retrieve

    Returns:
        A list of retrieved documents
    """
    logger.info(f"Retrieving documents for query: {query}")
    return vectordb.similarity_search(query, k=k)

def retrieve_documents_from_multiple_dbs(
    query: str,
    vectordbs: List[Any],
    k: int = 8,
    temp_db_boost: float = 2.0,
    batch_size: Optional[int] = None,
    retry_count: int = 1
) -> List[Document]:
    """
    Retrieve documents from multiple vector databases in parallel and prioritize temporary database results.
    Includes optimizations for batched processing and retry capability.

    Args:
        query: The query to search for
        vectordbs: List of vector databases to search
        k: Number of documents to retrieve from each database
        temp_db_boost: Boost factor for temporary database results (higher values prioritize temp DBs more)
        batch_size: Process databases in batches of this size (None = process all at once)
        retry_count: Number of times to retry failed retrievals

    Returns:
        Prioritized list of retrieved documents
    """
    if not vectordbs:
        logger.warning("No vector databases provided for retrieval")
        return []

    logger.info(f"Retrieving documents from {len(vectordbs)} databases in parallel")

    # Process in batches if specified and if there are enough databases
    if batch_size and batch_size > 0 and len(vectordbs) > batch_size:
        logger.info(f"Processing {len(vectordbs)} databases in batches of {batch_size}")
        all_docs_with_source = []

        # Process each batch
        for i in range(0, len(vectordbs), batch_size):
            batch = vectordbs[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(vectordbs) + batch_size - 1)//batch_size} ({len(batch)} databases)")

            # Use parallel processing to retrieve documents with source tracking
            batch_docs = parallel_document_retrieval(
                query,
                batch,
                k=k,
                timeout=30.0,  # Longer timeout for batch processing
                retry_count=retry_count
            )

            # Add batch results to overall results
            all_docs_with_source.extend(batch_docs)
    else:
        # Use parallel processing to retrieve documents with source tracking
        all_docs_with_source = parallel_document_retrieval(
            query,
            vectordbs,
            k=k,
            retry_count=retry_count
        )

    # Prioritize documents from temporary databases
    prioritized_docs = prioritize_documents(all_docs_with_source, temp_db_boost=temp_db_boost)

    logger.info(f"Retrieved and prioritized {len(prioritized_docs)} documents from {len(vectordbs)} databases")
    return prioritized_docs

class RagEngine:
    """
    Unified RAG engine that combines all RAG approaches.
    """

    def __init__(self,
                 reasoning_model_key: str,
                 tool_model_key: str,
                 vectordb = None):
        """
        Initialize the RAG engine.

        Args:
            reasoning_model_key: The key for the reasoning model
            tool_model_key: The key for the tool model
            vectordb: The vector database to use (defaults to the main vector store)
        """
        self.reasoning_model_key = reasoning_model_key
        self.tool_model_key = tool_model_key
        self.vectordb = vectordb or vector_store_manager.vectordb

    def query(self,
              user_query: str,
              use_reasoning: bool = True,
              use_web_search: bool = False,
              domain_context: str = "cyber_security",
              domain: str = "cyber_security",
              audience_level: str = "TECHNICAL",
              k: int = 8,
              chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Query the RAG engine with the given parameters.

        Args:
            user_query: The user's question to query the vector database with
            use_reasoning: Whether to use the reasoning model for analysis
            use_web_search: Whether to enhance results with web search
            domain_context: Domain context for web search
            domain: Domain for prompt templates
            audience_level: Technical level of the audience
            k: Number of documents to retrieve from the vector database
            chat_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]

        Returns:
            Dict containing the response and thought process
        """
        # Initialize thought process tracking
        thought_process = {
            "Retrieved Documents": [],
            "Document Sources": [],
            "Web Search Results": [],
            "Chat History Context": []
        }

        try:
            # Format chat history for context
            chat_context = ""
            if chat_history and len(chat_history) > 0:
                # Take last 5 messages for context
                recent_messages = chat_history[-5:]
                chat_context = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in recent_messages
                ])
                thought_process["Chat History Context"] = chat_context

            # Step 1: Retrieve documents from vector databases
            logger.info("Retrieving documents from vector databases")

            # Check if we have multiple vector databases in the session state
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'temp_vectordbs') and st.session_state.temp_vectordbs:
                # Combine permanent and temporary vector databases
                all_vectordbs = [self.vectordb] + st.session_state.temp_vectordbs
                logger.info(f"Using {len(all_vectordbs)} vector databases (1 permanent + {len(all_vectordbs) - 1} temporary)")

                # Get the current temp_db_boost value from session state if available
                current_temp_db_boost = config.temp_db_boost
                if hasattr(st.session_state, 'temp_db_boost'):
                    current_temp_db_boost = st.session_state.temp_db_boost
                    logger.info(f"Using temp_db_boost value from session state: {current_temp_db_boost}")
                else:
                    logger.info(f"Using default temp_db_boost value: {current_temp_db_boost}")

                # Determine batch size based on number of databases
                batch_size = None
                if len(all_vectordbs) > 5:
                    batch_size = 3  # Process in batches of 3 if we have more than 5 databases

                # Retrieve documents from all databases in parallel with prioritization
                all_docs = retrieve_documents_from_multiple_dbs(
                    query=user_query,
                    vectordbs=all_vectordbs,
                    k=k,
                    temp_db_boost=current_temp_db_boost,
                    batch_size=batch_size,
                    retry_count=1
                )

                # Log the distribution of documents by source
                temp_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'temp')
                perm_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'permanent')
                logger.info(f"Final document distribution: {temp_docs_count} from temporary DBs, {perm_docs_count} from permanent DB")
            else:
                # Just use the permanent vector database with caching
                logger.info("Using only the permanent vector database")
                all_docs = retrieve_documents(user_query, self.vectordb, k=k)

            # Track retrieved documents
            for doc in all_docs:
                thought_process["Retrieved Documents"].append(doc.page_content)
                thought_process["Document Sources"].append(doc.metadata.get("source", "Unknown"))

            # Step 2: Perform web search if enabled
            web_results = []
            if use_web_search:
                logger.info("Web search is enabled, retrieving web results")
                try:
                    # Map domain to appropriate search context
                    if domain == "senior_security_analyst":
                        search_context = "security analysis threat hunting"
                    else:
                        search_context = domain_context

                    web_results = search_web(user_query, domain_context=search_context)
                    thought_process["Web Search Results"] = web_results

                    # Add web results to temporary database
                    if web_results:
                        logger.info(f"Adding {len(web_results)} web search results to temporary database")
                        temp_db_result = web_results_to_temp_db(web_results)

                        if temp_db_result:
                            temp_vectordb, temp_db_dir = temp_db_result
                            logger.info(f"Successfully added web search results to temporary database at {temp_db_dir}")

                            # Re-run document retrieval with the new temporary database
                            if hasattr(st, 'session_state') and hasattr(st.session_state, 'temp_vectordbs'):
                                # Combine permanent and temporary vector databases
                                all_vectordbs = [self.vectordb] + st.session_state.temp_vectordbs
                                logger.info(f"Re-retrieving documents with {len(all_vectordbs)} vector databases")

                                # Get the current temp_db_boost value from session state if available
                                current_temp_db_boost = config.temp_db_boost
                                if hasattr(st.session_state, 'temp_db_boost'):
                                    current_temp_db_boost = st.session_state.temp_db_boost
                                    logger.info(f"Using temp_db_boost value from session state: {current_temp_db_boost}")
                                else:
                                    logger.info(f"Using default temp_db_boost value: {current_temp_db_boost}")

                                # Determine batch size based on number of databases
                                batch_size = None
                                if len(all_vectordbs) > 5:
                                    batch_size = 3  # Process in batches of 3 if we have more than 5 databases

                                # Retrieve documents from all databases in parallel with prioritization
                                all_docs = retrieve_documents_from_multiple_dbs(
                                    query=user_query,
                                    vectordbs=all_vectordbs,
                                    k=k,
                                    temp_db_boost=current_temp_db_boost,
                                    batch_size=batch_size,
                                    retry_count=1
                                )

                                # Log the distribution of documents by source
                                temp_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'temp')
                                perm_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'permanent')
                                logger.info(f"Final document distribution after web search: {temp_docs_count} from temporary DBs, {perm_docs_count} from permanent DB")

                                # Update retrieved documents in thought process
                                thought_process["Retrieved Documents"] = []
                                thought_process["Document Sources"] = []
                                for doc in all_docs:
                                    thought_process["Retrieved Documents"].append(doc.page_content)
                                    thought_process["Document Sources"].append(doc.metadata.get("source", "Unknown"))

                                logger.info(f"Retrieved {len(all_docs)} documents after adding web search results to temporary database")
                        else:
                            logger.warning("Failed to add web search results to temporary database")
                except Exception as e:
                    logger.error(f"Error in web search: {str(e)}")
                    thought_process["Web Search Error"] = str(e)

            # Step 3: Create context from retrieved documents and web results
            if not all_docs and not web_results:
                logger.warning("No relevant documents or web results found for query")
                return {
                    "response": "I couldn't find any relevant information in my knowledge base or web search. Please try a different query or provide more context.",
                    "thought_process": thought_process
                }

            # Combine document contents with efficient string building
            # Use a list to collect document contents and then join once
            doc_contents = []
            for doc in all_docs:
                # Add source information to help with attribution
                source_info = f"Source: {doc.metadata.get('source', 'Unknown')}"
                doc_contents.append(f"{doc.page_content}\n{source_info}")

            # Join all document contents at once
            context = "\n\n".join(doc_contents)

            # Calculate approximate token count for context
            approx_token_count = len(context) // 4  # Rough estimate: 4 chars per token
            logger.info(f"Document context approximate token count: {approx_token_count}")

            # Add web results if available
            web_context = ""
            if web_results:
                # Use a list to collect web snippets and then join once
                web_snippets = []
                for result in web_results:
                    # Format each web result
                    snippet = f"Title: {result.get('title', '')}\nURL: {result.get('link', '')}\nSnippet: {result.get('snippet', '')}"
                    web_snippets.append(snippet)

                # Join all web snippets at once
                web_context = "\n\n".join(web_snippets)

                # Calculate approximate token count for web context
                web_token_count = len(web_context) // 4  # Rough estimate: 4 chars per token
                logger.info(f"Web context approximate token count: {web_token_count}")

            # Format chat history context
            chat_context_formatted = chat_context if chat_context else "No previous conversation"

            # Combine contexts efficiently
            combined_context = "\n".join([
                "Previous Conversation:",
                chat_context_formatted,
                "",
                "Local Knowledge Base Results:",
                context,
                "",
                "Recent Web Search Results:" if web_context else "No web results available.",
                web_context
            ])

            # Step 4: Generate response based on the use_reasoning flag
            if use_reasoning:
                # First use reasoning model to analyze the context
                logger.info("Using reasoning model for analysis")

                # Create reasoning prompt using template
                reasoning_prompt = create_reasoning_prompt(
                    context=combined_context,
                    query=user_query,
                    domain=domain,
                    audience_level=audience_level,
                    chat_history=chat_context
                )
                thought_process["Reasoning Prompt"] = reasoning_prompt

                # Get analysis from reasoning model
                reasoning_response = call_reasoning_model(self.reasoning_model_key, reasoning_prompt)
                thought_process["Reasoning Analysis"] = reasoning_response

                # Then use tool model to generate final response based on reasoning
                logger.info("Using tool model for final response")

                # Create tool prompt using template
                tool_prompt = create_tool_prompt(
                    query=user_query,
                    reasoning_response=reasoning_response,
                    domain=domain,
                    audience_level=audience_level
                )
                thought_process["Tool Prompt"] = tool_prompt

                # Get final response from tool model
                tool_response = call_tool_model(self.tool_model_key, tool_prompt)
                thought_process["Tool Response"] = tool_response

                # Clean and return the response
                cleaned_response = clean_response(tool_response)

            else:
                # Direct chat mode - use tool model directly
                logger.info("Using direct chat mode")

                # Create direct prompt using template
                direct_prompt = create_direct_prompt(
                    context=combined_context,
                    query=user_query,
                    domain=domain,
                    audience_level=audience_level,
                    chat_history=chat_context
                )
                thought_process["Direct Prompt"] = direct_prompt

                # Get response from tool model
                direct_response = call_tool_model(self.tool_model_key, direct_prompt)
                thought_process["Direct Response"] = direct_response

                # Clean and return the response
                cleaned_response = clean_response(direct_response)

            # Return the final result
            return {
                "response": cleaned_response,
                "thought_process": thought_process
            }

        except Exception as e:
            logger.error(f"Error in RAG engine: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_response = f"I encountered an error while processing your request. Please try again or rephrase your question."
            return {
                "response": error_response,
                "thought_process": thought_process
            }

    def query_with_tools(self,
                         user_query: str,
                         domain: str = "cyber_security",
                         audience_level: str = "TECHNICAL",
                         use_web_search: bool = False,
                         chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Query the RAG engine using the enhanced tool-based approach.

        Args:
            user_query: The user's question
            domain: Domain for prompt templates
            audience_level: Technical level of the audience
            use_web_search: Whether to enable web search
            chat_history: List of previous messages

        Returns:
            Dict containing the response and thought process
        """
        # Initialize thought process tracking
        thought_process = {
            "Retrieved Documents": [],
            "Document Sources": [],
            "Web Search Results": [],
            "Reasoning Steps": [],
            "Self Reflection": None,
            "Final Analysis": None
        }

        try:
            # Format chat history for context
            chat_context = ""
            if chat_history and len(chat_history) > 0:
                # Take last 5 messages for context
                recent_messages = chat_history[-5:]
                chat_context = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in recent_messages
                ])
                thought_process["Chat History Context"] = chat_context

            # Step 1: Initial document retrieval
            logger.info("Retrieving documents from vector database")

            # Check if we have multiple vector databases in the session state
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'temp_vectordbs') and st.session_state.temp_vectordbs:
                # Combine permanent and temporary vector databases
                all_vectordbs = [self.vectordb] + st.session_state.temp_vectordbs
                logger.info(f"Using {len(all_vectordbs)} vector databases (1 permanent + {len(all_vectordbs) - 1} temporary)")

                # Get the current temp_db_boost value from session state if available
                current_temp_db_boost = config.temp_db_boost
                if hasattr(st.session_state, 'temp_db_boost'):
                    current_temp_db_boost = st.session_state.temp_db_boost
                    logger.info(f"Using temp_db_boost value from session state: {current_temp_db_boost}")
                else:
                    logger.info(f"Using default temp_db_boost value: {current_temp_db_boost}")

                # Determine batch size based on number of databases
                batch_size = None
                if len(all_vectordbs) > 5:
                    batch_size = 3  # Process in batches of 3 if we have more than 5 databases

                # Retrieve documents from all databases in parallel with prioritization
                all_docs = retrieve_documents_from_multiple_dbs(
                    query=user_query,
                    vectordbs=all_vectordbs,
                    k=5,
                    temp_db_boost=current_temp_db_boost,
                    batch_size=batch_size,
                    retry_count=1
                )

                # Log the distribution of documents by source
                temp_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'temp')
                perm_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'permanent')
                logger.info(f"Final document distribution: {temp_docs_count} from temporary DBs, {perm_docs_count} from permanent DB")
            else:
                # Just use the permanent vector database with caching
                logger.info("Using only the permanent vector database")
                all_docs = retrieve_documents(user_query, self.vectordb, k=5)

            # Track retrieved documents
            for doc in all_docs:
                thought_process["Retrieved Documents"].append(doc.page_content)
                thought_process["Document Sources"].append(doc.metadata.get("source", "Unknown"))

            # Step 2: Web search if enabled
            web_results = []
            if use_web_search:
                logger.info("Web search is enabled, retrieving web results")
                try:
                    # Map domain to appropriate search context
                    if domain == "senior_security_analyst":
                        search_context = "security analysis threat hunting"
                    else:
                        search_context = "cyber security"

                    web_results = search_web(user_query, domain_context=search_context)
                    thought_process["Web Search Results"] = web_results

                    # Add web results to temporary database
                    if web_results:
                        logger.info(f"Adding {len(web_results)} web search results to temporary database")
                        temp_db_result = web_results_to_temp_db(web_results)

                        if temp_db_result:
                            temp_vectordb, temp_db_dir = temp_db_result
                            logger.info(f"Successfully added web search results to temporary database at {temp_db_dir}")

                            # Re-run document retrieval with the new temporary database
                            if hasattr(st, 'session_state') and hasattr(st.session_state, 'temp_vectordbs'):
                                # Combine permanent and temporary vector databases
                                all_vectordbs = [self.vectordb] + st.session_state.temp_vectordbs
                                logger.info(f"Re-retrieving documents with {len(all_vectordbs)} vector databases")

                                # Get the current temp_db_boost value from session state if available
                                current_temp_db_boost = config.temp_db_boost
                                if hasattr(st.session_state, 'temp_db_boost'):
                                    current_temp_db_boost = st.session_state.temp_db_boost
                                    logger.info(f"Using temp_db_boost value from session state: {current_temp_db_boost}")
                                else:
                                    logger.info(f"Using default temp_db_boost value: {current_temp_db_boost}")

                                # Determine batch size based on number of databases
                                batch_size = None
                                if len(all_vectordbs) > 5:
                                    batch_size = 3  # Process in batches of 3 if we have more than 5 databases

                                # Retrieve documents from all databases in parallel with prioritization
                                all_docs = retrieve_documents_from_multiple_dbs(
                                    query=user_query,
                                    vectordbs=all_vectordbs,
                                    k=5,
                                    temp_db_boost=current_temp_db_boost,
                                    batch_size=batch_size,
                                    retry_count=1
                                )

                                # Log the distribution of documents by source
                                temp_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'temp')
                                perm_docs_count = sum(1 for doc in all_docs if doc.metadata.get('source_db') == 'permanent')
                                logger.info(f"Final document distribution after web search: {temp_docs_count} from temporary DBs, {perm_docs_count} from permanent DB")

                                # Update retrieved documents in thought process
                                thought_process["Retrieved Documents"] = []
                                thought_process["Document Sources"] = []
                                for doc in all_docs:
                                    thought_process["Retrieved Documents"].append(doc.page_content)
                                    thought_process["Document Sources"].append(doc.metadata.get("source", "Unknown"))

                                logger.info(f"Retrieved {len(all_docs)} documents after adding web search results to temporary database")
                        else:
                            logger.warning("Failed to add web search results to temporary database")
                except Exception as e:
                    logger.error(f"Error in web search: {str(e)}")
                    thought_process["Web Search Error"] = str(e)

            # Step 3: Create initial context
            if not all_docs and not web_results:
                logger.warning("No relevant documents or web results found for query")
                return {
                    "response": "I couldn't find any relevant information in my knowledge base or web search. Please try a different query or provide more context.",
                    "thought_process": thought_process
                }

            # Combine document contents with efficient string building
            # Use a list to collect document contents and then join once
            doc_contents = []
            for doc in all_docs:
                # Add source information to help with attribution
                source_info = f"Source: {doc.metadata.get('source', 'Unknown')}"
                doc_contents.append(f"{doc.page_content}\n{source_info}")

            # Join all document contents at once
            context = "\n\n".join(doc_contents)

            # Calculate approximate token count for context
            approx_token_count = len(context) // 4  # Rough estimate: 4 chars per token
            logger.info(f"Document context approximate token count: {approx_token_count}")

            # Add web results if available
            web_context = ""
            if web_results:
                # Use a list to collect web snippets and then join once
                web_snippets = []
                for result in web_results:
                    # Format each web result
                    snippet = f"Title: {result.get('title', '')}\nURL: {result.get('link', '')}\nSnippet: {result.get('snippet', '')}"
                    web_snippets.append(snippet)

                # Join all web snippets at once
                web_context = "\n\n".join(web_snippets)

                # Calculate approximate token count for web context
                web_token_count = len(web_context) // 4  # Rough estimate: 4 chars per token
                logger.info(f"Web context approximate token count: {web_token_count}")

            # Format chat history context
            chat_context_formatted = chat_context if chat_context else "No previous conversation"

            # Combine contexts efficiently
            combined_context = "\n".join([
                "Previous Conversation:",
                chat_context_formatted,
                "",
                "Local Knowledge Base Results:",
                context,
                "",
                "Recent Web Search Results:" if web_context else "No web results available.",
                web_context
            ])

            # Step 4: Create the reasoning prompt using the template
            reasoning_prompt = create_reasoning_prompt(
                context=combined_context,
                query=user_query,
                domain=domain,
                audience_level=audience_level,
                chat_history=chat_context
            )
            thought_process["Reasoning Prompt"] = reasoning_prompt

            # Step 5: Get reasoning response
            logger.info("Getting reasoning response")
            reasoning_response = call_reasoning_model(self.reasoning_model_key, reasoning_prompt)
            thought_process["Final Reasoning"] = reasoning_response

            # Step 6: Enhanced self-reflection and chain-of-thought reasoning
            try:
                reflection_prompt = f"""
                Review your reasoning process for the question: "{user_query}"

                Your reasoning:
                {reasoning_response}

                Follow this structured self-reflection process:

                1. REASONING ASSESSMENT
                - Identify the strongest parts of your analysis
                - Note any logical gaps or weak points in your reasoning
                - Highlight any assumptions you made that should be verified

                2. EVIDENCE EVALUATION
                - Assess if you've properly utilized all relevant evidence from the context
                - Identify any contradictions or inconsistencies in the evidence
                - Note if you've given appropriate weight to different pieces of evidence

                3. ALTERNATIVE PERSPECTIVES
                - Consider alternative interpretations of the evidence
                - Explore other possible conclusions that could be drawn
                - Think about how different expertise might approach this problem

                4. IMPROVEMENT PLAN
                - Specify what additional information would strengthen your analysis
                - Suggest how your reasoning could be more structured or comprehensive
                - Identify any technical details that should be elaborated further

                Provide a thorough self-reflection that will help improve the final analysis.
                """

                reflection_response = call_reasoning_model(self.reasoning_model_key, reflection_prompt)
                thought_process["Self Reflection"] = reflection_response

                # Incorporate reflection into final reasoning with explicit chain-of-thought
                enhanced_reasoning = f"""
                Original reasoning:
                {reasoning_response}

                Self-reflection and improvements:
                {reflection_response}

                Now, provide an enhanced analysis that addresses the gaps and improvements identified in the self-reflection.
                Follow this chain-of-thought process:

                1. REVISED UNDERSTANDING
                - Restate the core question with any clarifications needed
                - Incorporate insights from the self-reflection

                2. STRENGTHENED ANALYSIS
                - Address each gap or weakness identified in the self-reflection
                - Provide more detailed technical explanations where needed
                - Incorporate alternative perspectives when valuable

                3. CONFIDENCE ASSESSMENT
                - Indicate confidence levels for different parts of your analysis
                - Explicitly state remaining uncertainties or limitations

                4. FINAL COMPREHENSIVE ANALYSIS
                - Synthesize all information into a cohesive, improved analysis
                - Ensure all aspects of the original query are thoroughly addressed
                - Provide specific, actionable insights based on the available evidence

                Provide your enhanced analysis:
                """

                # Get enhanced reasoning with improved chain-of-thought
                enhanced_response = call_reasoning_model(self.reasoning_model_key, enhanced_reasoning)
                thought_process["Enhanced Reasoning"] = enhanced_response

                # Use the enhanced reasoning as the final reasoning
                reasoning_response = enhanced_response
            except Exception as e:
                logger.error(f"Error in self-reflection step: {str(e)}")
                # Continue with original reasoning if reflection fails

            # Step 7: Pass the reasoning to the tool model for final response
            logger.info("Creating prompt for tool model")
            tool_prompt = create_tool_prompt(
                query=user_query,
                reasoning_response=reasoning_response,
                domain=domain,
                audience_level=audience_level
            )
            thought_process["Tool Prompt"] = tool_prompt

            # Step 8: Get response from tool model
            logger.info("Sending prompt to tool model")
            tool_response = call_tool_model(self.tool_model_key, tool_prompt)
            thought_process["Tool Response"] = tool_response

            # Step 9: Clean and return the final result
            cleaned_response = clean_response(tool_response)
            final_result = {
                "response": cleaned_response,
                "thought_process": thought_process
            }

            return final_result

        except Exception as e:
            logger.error(f"Error in RAG engine with tools: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_response = f"I encountered an error while processing your request. Please try again or rephrase your question."
            return {
                "response": error_response,
                "thought_process": thought_process
            }

# Create a singleton instance
rag_engine = RagEngine(
    reasoning_model_key=None,  # Will be set after model initialization
    tool_model_key=None,       # Will be set after model initialization
    vectordb=None              # Will use the default vector store
)

def initialize_rag_engine(reasoning_model_key: str, tool_model_key: str) -> None:
    """
    Initialize the RAG engine with model keys.

    Args:
        reasoning_model_key: The key for the reasoning model
        tool_model_key: The key for the tool model
    """
    global rag_engine
    rag_engine = RagEngine(
        reasoning_model_key=reasoning_model_key,
        tool_model_key=tool_model_key,
        vectordb=vector_store_manager.vectordb
    )
    logger.info("RAG engine initialized with model keys")

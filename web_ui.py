"""
Web UI for CTI Agent.
Provides a Streamlit interface for interacting with the RAG system.
"""

import streamlit as st
import re
import tempfile
import os
import shutil
import atexit
import logging
import time
import torch
from typing import Dict, Any, List, Optional
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from utils.common import setup_logging, get_device, suppress_warnings
from model_manager import initialize_models
from rag_engine import initialize_rag_engine, rag_engine
from document_processor import process_pdf, process_website, create_temp_vector_store, cleanup_vector_store
from vector_store import vector_store_manager
from response_cleaner import clean_response

# Set up logging and suppress warnings
setup_logging(config.log_level)
suppress_warnings()  # Explicitly call suppress_warnings to handle all warnings
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(
    page_title="Oracle - CTI Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Oracle CTI Agent\nPowered by AgenticRAG"
    }
)

# Apply minimalistic styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Clean up header */
    header {
        background-color: transparent !important;
        border-bottom: none !important;
    }
    
    /* Improve chat container */
    [data-testid="stChatMessageContent"] {
        padding: 0.5rem 0;
    }
    
    /* Make chat input stand out */
    [data-testid="stChatInput"] {
        background-color: white;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Load custom CSS
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply custom styling
load_css()

def init_chat_history():
    """Initialize chat history in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def handle_file_upload():
    """Handle file upload in the sidebar."""
    # Initialize processed files tracking in session state if it doesn't exist
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

    uploaded_file = st.file_uploader(
        label="Upload a PDF file",
        type=['pdf'],
        key="pdf_file_uploader"
    )

    if uploaded_file is not None:
        # Create a unique identifier for this file (name + size + last_modified)
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # Check if this file has already been processed
        if file_id in st.session_state.processed_files:
            # File already processed, no need to process again
            logger.info(f"File {uploaded_file.name} already processed, skipping")
            return

        with st.spinner("Processing uploaded file..."):
            # Create a temporary file to store the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Process the uploaded PDF file
                logger.info(f"Processing PDF file: {tmp_path}")
                chunks = process_pdf(tmp_path)

                if not chunks:
                    st.error("No content could be extracted from the file. The PDF might be empty, scanned, or protected.")
                    logger.warning(f"No chunks were extracted from {uploaded_file.name}")
                    return

                logger.info(f"Successfully extracted {len(chunks)} chunks from PDF")

                try:
                    # Create temporary vector store for this session
                    temp_vectordb, temp_db_dir = vector_store_manager.create_temp_vector_store(chunks)

                    # Store the vectordb in session state
                    if 'temp_vectordbs' not in st.session_state:
                        st.session_state.temp_vectordbs = []
                    st.session_state.temp_vectordbs.append(temp_vectordb)

                    # Mark this file as processed
                    st.session_state.processed_files.add(file_id)

                    st.success(f"Successfully processed {uploaded_file.name} ({len(chunks)} chunks created)")

                    # Initialize messages list if it doesn't exist
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Add system message about the processed document
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"New document '{uploaded_file.name}' has been processed and is ready for questions. Note: This document will only be available for the current session."
                    })
                except Exception as vector_error:
                    logger.error(f"Error creating vector store: {str(vector_error)}")
                    st.error(f"Error creating vector store: {str(vector_error)}")
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                st.error(f"Error processing file: {str(e)}")
            finally:
                # Clean up temporary PDF file
                try:
                    os.unlink(tmp_path)
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")

def handle_url_input():
    """Handle website URL input in the sidebar."""
    # Initialize processed websites tracking in session state if it doesn't exist
    if 'processed_websites' not in st.session_state:
        st.session_state.processed_websites = set()

    url = st.text_input(
        label="Enter a website URL to analyze",
        key="website_url_input"
    )

    if url and st.button("Process Website", key="process_website_button"):
        # Check if this website has already been processed
        if url in st.session_state.processed_websites:
            # Website already processed, no need to process again
            logger.info(f"Website {url} already processed, skipping")
            return

        with st.spinner("Scraping and processing website content..."):
            try:
                # Process the website
                logger.info(f"Processing website: {url}")
                chunks = process_website(url)

                if not chunks:
                    logger.error("No chunks were created from the website content")
                    st.error("No content could be extracted from the website")
                    return

                logger.info(f"Successfully extracted {len(chunks)} chunks from website")

                try:
                    # Create temporary vector store for this session
                    temp_vectordb, temp_db_dir = vector_store_manager.create_temp_vector_store(chunks)

                    # Store the vectordb in session state
                    if 'temp_vectordbs' not in st.session_state:
                        st.session_state.temp_vectordbs = []
                    st.session_state.temp_vectordbs.append(temp_vectordb)

                    # Mark this website as processed
                    st.session_state.processed_websites.add(url)

                    # Log success
                    logger.info(f"Successfully processed website: {url}")
                    logger.info(f"Created {len(chunks)} chunks from the website content")

                    st.success(f"Successfully processed website: {url}")

                    # Initialize messages list if it doesn't exist
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Add system message about the processed website
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Website '{url}' has been processed and is ready for questions. Created {len(chunks)} chunks from the content."
                    })

                except Exception as e:
                    error_msg = f"Error creating vector store: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    return

            except Exception as e:
                error_msg = f"Error processing website: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)

def display_sidebar():
    """Display the sidebar with configuration options."""
    with st.sidebar:
        # Clean, minimalistic header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">Oracle 🔮</h1>
            <p style="font-size: 1rem; opacity: 0.8;">CTI Analysis with AgenticRAG</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a section for Reasoning with an emoji and improved description
        st.markdown("### Reasoning Optimization")

        # Get the reasoning model from environment variable
        reasoning_model = os.getenv("REASONING_MODEL_ID", "default reasoning model")

        # Add more detailed explanation of reasoning mode
        st.markdown(f"""
        When enabled, the system uses a specialized reasoning model to analyze information before generating a response.
        """)

        # Toggle for reasoning with improved help text
        use_reasoning = st.toggle(
            "Enable Advanced Reasoning",
            value=st.session_state.get('use_reasoning', True),
            key="reasoning_toggle",
            help="Enables enhanced reasoning and chain-of-thought analysis"
        )

        # Update the session state and config if the value changed
        if use_reasoning != st.session_state.get('use_reasoning', True):
            st.session_state.use_reasoning = use_reasoning

            # If reasoning is enabled, use the REASONING_MODEL_ID from .env
            if use_reasoning:
                # Store current model ID for logging purposes
                previous_model = config.reasoning_model_id
                # Use the original REASONING_MODEL_ID from .env file
                config.reasoning_model_id = os.getenv("REASONING_MODEL_ID", config.reasoning_model_id)

                logger.info(f"Switching reasoning model from {previous_model} to {config.reasoning_model_id}")
                st.info(f"Advanced reasoning enabled. Using {config.reasoning_model_id} for enhanced analysis and reasoning.")

                # Reinitialize models with the new reasoning model ID
                try:
                    model_keys = initialize_models(config.reasoning_model_id, config.tool_model_id, config.huggingface_api_token)
                    initialize_rag_engine(model_keys['reasoning'], model_keys['tool'])
                    st.success(f"Reasoning model successfully initialized.")
                except Exception as e:
                    st.error(f"Error initializing reasoning model: {str(e)}")
                    logger.error(f"Error initializing reasoning model: {str(e)}")
            else:
                # If reasoning is disabled, revert to the selected model for direct chat
                if 'selected_model' in st.session_state:
                    previous_model = config.reasoning_model_id
                    config.reasoning_model_id = st.session_state.selected_model

                    logger.info(f"Switching from reasoning model {previous_model} to direct chat model {config.reasoning_model_id}")
                    st.info(f"Advanced reasoning disabled. Using {st.session_state.selected_model} for direct responses (faster but less thorough).")

                    # Reinitialize models with the selected model
                    try:
                        model_keys = initialize_models(config.reasoning_model_id, config.tool_model_id, config.huggingface_api_token)
                        initialize_rag_engine(model_keys['reasoning'], model_keys['tool'])
                        st.success(f"Direct chat model successfully initialized.")
                    except Exception as e:
                        st.error(f"Error initializing direct chat model: {str(e)}")
                        logger.error(f"Error initializing direct chat model: {str(e)}")

        # Add Domain Role Selector with improved descriptions
        st.markdown("### 🧪 Domain Role")

        domain_role_options = {
            "cyber_security": "Cyber Threat Intelligence (CTI) Analyst",
            "senior_security_analyst": "Senior Security Analyst"
        }

        # Initialize domain role in session state if not present
        if 'domain_role' not in st.session_state:
            st.session_state.domain_role = "cyber_security"

        # Add domain role selector
        selected_domain_role = st.selectbox(
            label="Select Domain Role",
            options=list(domain_role_options.keys()),
            format_func=lambda x: domain_role_options[x],
            key="domain_role_select",
            help="Select the domain expertise role for the AI assistant"
        )

        # Add enhanced role descriptions
        if selected_domain_role == "cyber_security":
            st.caption("""
            **Cyber Threat Intelligence (CTI) Analyst**:
            - Focuses on threat intelligence analysis and reporting
            - Specializes in malware analysis, threat actor profiling, and IOC identification
            - Best for general CTI questions and threat analysis
            """)
        else:
            st.caption("""
            **Senior Security Analyst**:
            - Advanced expertise in security operations and threat hunting
            - Specialized knowledge in Sysmon, eBPF, AzureEntra ID, Sentinel, CrowdStrike,
              Velociraptor, Zeek, Suricata, PaloAlto, Okta, ZScalar
            - Strong focus on Detection Engineering and Usecase Development
            - Best for technical security operations questions and advanced threat hunting
            """)

        # Update the session state with the selected domain role
        if selected_domain_role != st.session_state.domain_role:
            st.session_state.domain_role = selected_domain_role
            st.info(f"Domain role changed to {domain_role_options[selected_domain_role]}. This will affect the expertise and focus of responses.")

        # Add Audience Level Selector
        st.markdown("### 👥 Audience Level")

        audience_options = {
            "EXECUTIVE": "Executive (C-Suite, Leadership)",
            "TECHNICAL": "Technical (Senior Analysts, Engineers)",
            "OPERATIONAL": "Operational (Team Leads, Managers)",
            "JUNIOR": "Junior (Analysts, New Team Members)"
        }

        # Initialize audience level in session state if not present
        if 'audience_level' not in st.session_state:
            st.session_state.audience_level = "TECHNICAL"

        # Add audience level selector
        selected_audience = st.selectbox(
            label="Select Audience Level",
            options=list(audience_options.keys()),
            format_func=lambda x: audience_options[x],
            key="audience_level_select",
            help="Adjust the technical depth and focus of the analysis based on the target audience"
        )

        # Update the session state with the selected audience level
        st.session_state.audience_level = selected_audience

        # Add Model Selection section
        st.markdown("### 🤖 Model Selection")

        # Initialize model selection in session state if not present
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "llama3:8b-64k"  # Default model

        # Add model selection radio buttons with improved descriptions
        selected_model = st.radio(
            "Select Model",
            options=["llama3:8b-64k", "qwen2.5:7b-64k"],
            index=0 if st.session_state.selected_model == "llama3:8b-64k" else 1,
            key="model_selection_radio",
            help="Select the model to use for responses and direct chat mode"
        )

        # Add enhanced model descriptions with more details
        if selected_model == "llama3:8b-64k":
            st.caption("""
            **Llama3 8B**: Optimized for general RAG and comprehensive reasoning
            - Better for general questions and comprehensive analysis
            - Stronger contextual understanding and reasoning
            - Recommended for most general security questions
            """)
        else:
            st.caption("""
            **Qwen2.5 7B**: Specialized for technical analysis and IOC extraction
            - Better at identifying and extracting technical indicators
            - More precise with technical details and patterns
            - Recommended for IOC extraction and technical threat analysis
            """)

        # Update the session state and config if the value changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model

            # Update the tool model IDs based on selection
            if selected_model == "llama3:8b-64k":
                config.general_tool_model_id = "llama3:8b-64k"
                st.info(f"Switched to Llama3 model for general RAG and comprehensive reasoning.")
            else:
                config.general_tool_model_id = "qwen2.5:7b-64k"
                st.info(f"Switched to Qwen2.5 model for improved IOC extraction and technical analysis.")

            # Set the tool model based on current mode
            if st.session_state.get('ioc_extraction_mode', False):
                config.tool_model_id = config.ioc_tool_model_id
                st.info(f"Using {config.ioc_tool_model_id} for final responses (IOC Extraction Mode is enabled).")
            else:
                config.tool_model_id = config.general_tool_model_id
                st.info(f"Using {config.general_tool_model_id} for final responses.")

            # Only update the reasoning model if reasoning is disabled
            # If reasoning is enabled, we always use the model from REASONING_MODEL_ID in .env
            if not st.session_state.get('use_reasoning', True):
                config.reasoning_model_id = selected_model
                st.info(f"Reasoning mode is disabled, using {selected_model} for direct chat.")
            else:
                # If reasoning is enabled, use the REASONING_MODEL_ID from .env
                config.reasoning_model_id = os.getenv("REASONING_MODEL_ID", config.reasoning_model_id)
                st.info(f"Reasoning mode is enabled, using {config.reasoning_model_id} for reasoning.")

            # Reinitialize models with the new model IDs
            try:
                model_keys = initialize_models(config.reasoning_model_id, config.tool_model_id, config.huggingface_api_token)
                initialize_rag_engine(model_keys['reasoning'], model_keys['tool'])
                st.success(f"Models successfully reinitialized.")
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
                logger.error(f"Error initializing models: {str(e)}")

        # Add IOC extraction mode toggle
        st.markdown("### 🔍 IOC Extraction Mode")
        st.write("🔎 Optimize for Indicators of Compromise")

        # Initialize IOC extraction mode in session state if not present
        if 'ioc_extraction_mode' not in st.session_state:
            st.session_state.ioc_extraction_mode = config.ioc_extraction_mode

        # Add IOC extraction mode toggle
        ioc_extraction_mode = st.toggle(
            "Enable IOC Extraction Mode",
            value=st.session_state.ioc_extraction_mode,
            key="ioc_extraction_toggle",
            help="Toggle between general RAG and IOC extraction optimized mode"
        )

        # Update the session state and config if the value changed
        if ioc_extraction_mode != st.session_state.ioc_extraction_mode:
            st.session_state.ioc_extraction_mode = ioc_extraction_mode
            config.ioc_extraction_mode = ioc_extraction_mode

            # Update the tool model ID based on the IOC extraction mode
            if ioc_extraction_mode:
                config.tool_model_id = config.ioc_tool_model_id
                st.info(f"IOC Extraction Mode enabled. Using {config.ioc_tool_model_id} for final responses.")
            else:
                config.tool_model_id = config.general_tool_model_id
                st.info(f"IOC Extraction Mode disabled. Using {config.general_tool_model_id} for final responses.")

            # Keep the reasoning model from .env if reasoning is enabled
            if st.session_state.get('use_reasoning', True):
                config.reasoning_model_id = os.getenv("REASONING_MODEL_ID", config.reasoning_model_id)
                st.info(f"Reasoning mode is enabled, using {config.reasoning_model_id} for reasoning.")

            # Reinitialize models with the new tool model ID
            model_keys = initialize_models(config.reasoning_model_id, config.tool_model_id, config.huggingface_api_token)
            initialize_rag_engine(model_keys['reasoning'], model_keys['tool'])

        # Add file upload section in sidebar
        st.subheader("Upload Documents")
        handle_file_upload()

        # Add website URL input section
        st.subheader("Process Website")
        handle_url_input()

        # Add Temporary Database Boost slider - always visible
        st.markdown("### 📊 Database Prioritization")
        st.write("Adjust the weight distribution between temporary and permanent databases:")

        # Add slider for temp_db_boost
        temp_db_boost = st.slider(
            "Weight Distribution (Temp vs Permanent)",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.temp_db_boost,
            step=0.1,
            format="%.1f",
            key="temp_db_boost_slider",
            help="Higher values prioritize temporary database results (processed chunks) more. Default is 2.0."
        )

        # Update the config and session state if the value changed
        if temp_db_boost != st.session_state.temp_db_boost:
            st.session_state.temp_db_boost = temp_db_boost
            config.temp_db_boost = temp_db_boost
            st.info(f"Temporary database boost factor updated to {temp_db_boost}. This will affect how much priority is given to processed website content in responses.")

        # Show status of temporary databases
        if 'temp_vectordbs' in st.session_state and st.session_state.temp_vectordbs:
            st.success(f"✅ Temporary database active with {len(st.session_state.temp_vectordbs)} sources")
        else:
            st.warning("⚠️ No temporary database active. Process a website or upload a document to create one.")

        # Web Search section with toggle
        st.write("🌐 Web Search")

        # Initialize session state variables if they don't exist
        if 'web_search_enabled' not in st.session_state:
            st.session_state.web_search_enabled = config.web_search_enabled

        # Web search toggle
        st.session_state.web_search_enabled = st.toggle(
            "Enable Web Search",
            value=st.session_state.web_search_enabled,
            key="web_search_toggle",
            help="Enable real-time web search enhancement"
        )

        # Search provider selection
        if st.session_state.web_search_enabled:
            search_provider_options = {
                "serpapi": "SerpAPI (Google Search)",
                "duckduckgo": "DuckDuckGo"
            }

            # Default to SerpAPI unless DuckDuckGo is explicitly selected
            selected_provider = st.selectbox(
                "Search Provider",
                options=list(search_provider_options.keys()),
                format_func=lambda x: search_provider_options[x],
                index=0 if config.search_provider != "duckduckgo" else 1,
                key="search_provider_select"
            )

            # Update session state and environment variable
            if selected_provider != config.search_provider:
                config.search_provider = selected_provider
                os.environ["SEARCH_PROVIDER"] = selected_provider

            # Show API key input for SerpAPI
            if selected_provider == "serpapi":
                serpapi_key = st.text_input(
                    "SerpAPI Key",
                    value=config.serpapi_key,
                    type="password",
                    key="serpapi_key_input",
                    help="Enter your SerpAPI key. Get one at https://serpapi.com/"
                )

                # Update environment variable if key changed
                if serpapi_key and serpapi_key != config.serpapi_key:
                    config.serpapi_key = serpapi_key
                    os.environ["SERPAPI_KEY"] = serpapi_key

        if st.button("Clear Chat History", key="clear_chat_button"):
            st.session_state.messages = []
            vector_store_manager.cleanup_temp_vector_stores()
            if 'temp_vectordbs' in st.session_state:
                st.session_state.temp_vectordbs = []
            # Clear processed files and websites tracking
            st.session_state.processed_files = set()
            st.session_state.processed_websites = set()

        # Add empty space to push content to the bottom
        st.markdown("##")
        st.markdown("##")

        # Add contributor section at the bottom of the sidebar
        with st.container():
            st.markdown("""
            <div class="contributor-section">
                <p><strong>Developed by:</strong> Dexterlabs</p>
                <p><strong>Author:</strong> Dhruv Majumdar</p>
            </div>
            """, unsafe_allow_html=True)

def display_chat_history():
    """Display the chat history."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # For assistant messages, display thinking process if available
            if message["role"] == "assistant" and "thought_process" in message:
                thought_process = message["thought_process"]

                # Create an expander for the thinking process
                with st.expander("View Reasoning Process 🤔"):
                    st.markdown("### Model's Thought Process")

                    # Display retrieved documents if available
                    if "Retrieved Documents" in thought_process and thought_process["Retrieved Documents"]:
                        st.markdown("#### 📚 Retrieved Context")
                        docs = thought_process["Retrieved Documents"]
                        sources = thought_process.get("Document Sources", ["Unknown"] * len(docs))

                        # Display documents
                        for i, (doc, source) in enumerate(zip(docs, sources), 1):
                            st.markdown(f"**Document {i} - {source}**")
                            st.markdown(doc)
                            st.markdown("---")  # Add a separator between documents

                    # Display prompts and responses in order
                    display_order = [
                        ("Reasoning Prompt", "🧩 Reasoning Prompt"),
                        ("Reasoning Analysis", "🧠 Reasoning Analysis"),
                        ("Self Reflection", "🤔 Self Reflection"),
                        ("Enhanced Reasoning", "✨ Enhanced Reasoning"),
                        ("Tool Prompt", "🔧 Tool Prompt"),
                        ("Tool Response", "🔨 Tool Response"),
                        ("Direct Prompt", "💬 Direct Prompt"),
                        ("Direct Response", "📝 Direct Response"),
                        ("Final Reasoning", "🔬 Final Reasoning")
                    ]

                    for key, title in display_order:
                        if key in thought_process and thought_process[key]:
                            st.markdown(f"#### {title}")
                            st.markdown(f"```\n{thought_process[key]}\n```")

def handle_user_input(prompt: str):
    """Handle user input and generate a response."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get configuration from session state
            use_reasoning = st.session_state.get('use_reasoning', True)
            web_search_enabled = st.session_state.get('web_search_enabled', False)
            audience_level = st.session_state.get('audience_level', "TECHNICAL")
            domain_role = st.session_state.get('domain_role', "cyber_security")
            ioc_extraction_mode = st.session_state.get('ioc_extraction_mode', False)

            # Update the config's temp_db_boost with the current session state value
            if 'temp_db_boost' in st.session_state:
                config.temp_db_boost = st.session_state.temp_db_boost

            # Update the config's ioc_extraction_mode with the current session state value
            config.ioc_extraction_mode = ioc_extraction_mode

            # Get chat history
            chat_history = st.session_state.messages[:-1]  # Exclude current message

            # Query the RAG engine
            if use_reasoning:
                # Use the tool-based approach for more complex reasoning
                result = rag_engine.query_with_tools(
                    user_query=prompt,
                    domain=domain_role,
                    audience_level=audience_level,
                    use_web_search=web_search_enabled,
                    chat_history=chat_history
                )
            else:
                # Use the standard approach for direct chat
                result = rag_engine.query(
                    user_query=prompt,
                    use_reasoning=False,
                    use_web_search=web_search_enabled,
                    domain=domain_role,
                    audience_level=audience_level,
                    chat_history=chat_history
                )

            # Get the response
            response = result["response"]
            thought_process = result.get("thought_process", {})

            # Add to chat history with thought process
            message_to_add = {
                "role": "assistant",
                "content": response,
                "thought_process": thought_process
            }
            st.session_state.messages.append(message_to_add)

            # Display the response
            st.markdown(response)

def initialize_session_state():
    """Initialize session state variables."""
    # Initialize web search settings
    if 'web_search_enabled' not in st.session_state:
        st.session_state.web_search_enabled = config.web_search_enabled

    # Initialize search provider
    if 'search_provider' not in st.session_state:
        st.session_state.search_provider = config.search_provider

    # Initialize reasoning mode
    if 'use_reasoning' not in st.session_state:
        st.session_state.use_reasoning = True

    # Initialize domain role
    if 'domain_role' not in st.session_state:
        st.session_state.domain_role = "cyber_security"

    # Initialize audience level
    if 'audience_level' not in st.session_state:
        st.session_state.audience_level = "TECHNICAL"

    # Initialize model selection
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "llama3:8b-64k"  # Default model

    # Initialize IOC extraction mode
    if 'ioc_extraction_mode' not in st.session_state:
        st.session_state.ioc_extraction_mode = config.ioc_extraction_mode

    # Initialize temp_db_boost
    if 'temp_db_boost' not in st.session_state:
        st.session_state.temp_db_boost = config.temp_db_boost

    # Initialize processed files tracking
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

    # Initialize processed websites tracking
    if 'processed_websites' not in st.session_state:
        st.session_state.processed_websites = set()

def cleanup():
    """Clean up resources when the application exits."""
    vector_store_manager.cleanup_temp_vector_stores()
    # Clear session state tracking variables
    if 'processed_files' in st.session_state:
        st.session_state.processed_files = set()
    if 'processed_websites' in st.session_state:
        st.session_state.processed_websites = set()

def main():
    """Main application entry point."""
    # Initialize session state first to ensure we have the model selection
    initialize_session_state()

    # Initialize chat history
    init_chat_history()

    # Set the reasoning model based on reasoning toggle and selected model
    if 'selected_model' in st.session_state:
        # Set the general tool model based on selected model
        config.general_tool_model_id = st.session_state.selected_model

        # If reasoning is enabled, use the REASONING_MODEL_ID from .env
        if st.session_state.get('use_reasoning', True):
            config.reasoning_model_id = os.getenv("REASONING_MODEL_ID", config.reasoning_model_id)
            logger.info(f"Reasoning mode is enabled, using {config.reasoning_model_id} for reasoning")
        else:
            # If reasoning is disabled, use the selected model
            config.reasoning_model_id = st.session_state.selected_model
            logger.info(f"Reasoning mode is disabled, using {st.session_state.selected_model} for direct chat")

    # Set the tool model ID based on IOC extraction mode
    if config.ioc_extraction_mode:
        tool_model_id = config.ioc_tool_model_id
    else:
        tool_model_id = config.general_tool_model_id

    # Initialize models
    model_keys = initialize_models(config.reasoning_model_id, tool_model_id, config.huggingface_api_token)

    # Initialize RAG engine
    initialize_rag_engine(model_keys['reasoning'], model_keys['tool'])

    # Display sidebar
    display_sidebar()

    # Display chat interface
    display_chat_history()

    # Chat input
    if prompt := st.chat_input(
        placeholder="Ask a security question...",
        key="chat_input"
    ):
        handle_user_input(prompt)

    # Register cleanup function
    atexit.register(cleanup)

if __name__ == "__main__":
    main()

# 🔮 Oracle: CTI Agentic RAG

<div align="center">
  <img src="./screenshots/oracle_logo.png" alt="Oracle Logo" width="200"/>
  <p><em>Advanced Cyber Threat Intelligence Analysis with Agentic RAG</em></p>
</div>

## 📋 Overview

Oracle is a sophisticated Retrieval-Augmented Generation (RAG) system with agent capabilities specifically designed for Cyber Threat Intelligence (CTI) analysis. It combines the power of large language models with specialized knowledge retrieval to provide accurate, contextual, and actionable intelligence for security professionals.

### Key Features

- **🧠 Agentic RAG Architecture**: Uses a reasoning model for analysis followed by a specialized tool model for final responses
- **🌐 Web-Enhanced Intelligence**: Augments local knowledge with real-time web search results
- **📊 Multi-Database Prioritization**: Intelligently balances between permanent and temporary knowledge bases
- **🔍 Specialized CTI Analysis**: Optimized for cyber security domain with IOC extraction capabilities
- **📝 Document Processing**: Handles PDFs, text files, JSON, and website content
- **💻 Intuitive Web Interface**: Clean, minimalistic UI built with Streamlit

<div align="center">
  <img src="./screenshots/oracle_overview.png" alt="Oracle Overview" width="800"/>
  <p><em>Oracle's key components and workflow</em></p>
</div>

## 🏗️ Architecture

Oracle uses a sophisticated architecture that combines multiple components to deliver powerful CTI analysis capabilities:

<div align="center">
  <img src="./screenshots/architecture_diagram.png" alt="Architecture Diagram" width="800"/>
  <p><em>Oracle's component architecture</em></p>
</div>

## 🚀 Installation

### Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/) (recommended for environment management)
- [Ollama](https://ollama.ai) (for local model inference)

### Environment Setup

1. **Create and activate a new conda environment:**

```bash
conda create -n oracle-cti python=3.11
conda activate oracle-cti
```

2. **Clone the repository:**

```bash
git clone https://github.com/yourusername/oracle-cti.git
cd oracle-cti
```

3. **Install required packages:**

```bash
python -m pip install -r requirements.txt
pip install 'smolagents[openai]'
```

<div align="center">
  <img src="./screenshots/installation.png" alt="Installation Process" width="700"/>
  <p><em>Terminal view of the installation process</em></p>
</div>

### Ollama Setup (for Local Inference)

1. **Install Ollama** from [ollama.ai](https://ollama.ai)

2. **Pull the base models:**

```bash
ollama pull huihui_ai/deepseek-r1-abliterated:14b-qwen-distill-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull llama3:8b-instruct-q4_K_M
```

3. **Create custom models with extended context windows:**

```bash
# Create Deepseek model with 64k context - recommended for reasoning
ollama create deepseek-r1:14b-64k-ab -f ollama_models/Deepseek-r1-14b-64k-ab

# Create Qwen model with 64k context - recommended for CTI purpose
ollama create qwen2.5:7b-64k -f ollama_models/Qwen-7b-Instruct-64K

# Create Llama model with 64k context - recommended for general RAG
ollama create llama3:8b-64k -f ollama_models/Llama-3-8b-64k
```

<div align="center">
  <img src="./screenshots/ollama_setup.png" alt="Ollama Setup" width="700"/>
  <p><em>Setting up Ollama with custom models</em></p>
</div>

> **Note**: You can also use the provided `rebuild_models.sh` script to create all required models:
> ```bash
> chmod +x rebuild_models.sh
> ./rebuild_models.sh
> ```

## ⚙️ Configuration

Create a `.env` file in the project root with your configuration settings:

### Local Inference with Ollama

```env
# Model Configuration
USE_HUGGINGFACE=no
HUGGINGFACE_API_TOKEN=
REASONING_MODEL_ID=deepseek-r1:14b-64k-ab
TOOL_MODEL_ID=qwen2.5:7b-64k
GENERAL_TOOL_MODEL_ID=llama3:8b-64k
IOC_TOOL_MODEL_ID=qwen2.5:7b-64k

# Web Search Configuration
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMELIMIT=y

# Search Provider Configuration
# Options: duckduckgo, serpapi
SEARCH_PROVIDER=serpapi

# SerpAPI Configuration
# Get your API key from https://serpapi.com/
SERPAPI_KEY=your_serpapi_key_here

# Temporary Database Configuration
# Higher values prioritize temporary database results more
# Default is 2.0, which means temporary database results are weighted twice as much
TEMP_DB_BOOST=2.0
```

<div align="center">
  <img src="./screenshots/env_config.png" alt="Environment Configuration" width="700"/>
  <p><em>Example .env configuration file</em></p>
</div>

### Cloud API with HuggingFace (Alternative)

```env
USE_HUGGINGFACE=yes
HUGGINGFACE_API_TOKEN=your_token_here
REASONING_MODEL_ID=deepseek-r1:14b-64k-ab
TOOL_MODEL_ID=qwen2.5:7b-64k
```

## 📊 Usage

### Admin: Creating Permanent Vector Databases

The `ingest.py` script is used to create permanent vector databases from various data sources:

```bash
python ingest.py
```

This script will:
1. Process all new files in the `data` directory
2. Create or update the vector database in the `chroma_db` directory
3. Prompt for website URLs to process (optional)

<div align="center">
  <img src="./screenshots/ingest_process.png" alt="Ingest Process" width="700"/>
  <p><em>Running the ingest.py script to create vector databases</em></p>
</div>

#### Data Directory Structure

Place your data files in the `data` directory with the following structure:

```
data/
  ├── pdf_files/
  │   ├── document1.pdf
  │   └── document2.pdf
  ├── text_files/
  │   ├── document3.txt
  │   └── document4.txt
  └── json_files/
      ├── document5.json
      └── document6.json
```

### End-User: Launching the Application

To start the application with optimized settings:

```bash
# Use the provided script
chmod +x start_optimized.sh
./start_optimized.sh

# Or run directly
export OLLAMA_FLASH_ATTENTION=1
python -m streamlit run web_ui.py
```

<div align="center">
  <img src="./screenshots/oracle_ui.png" alt="Oracle Web UI" width="800"/>
  <p><em>Oracle's intuitive web interface</em></p>
</div>

## 🖥️ Web UI Features

### Reasoning Optimization

- **Enable Advanced Reasoning**: When enabled, the system uses a specialized reasoning model to analyze information before generating a response.
  - **ON**: Uses structured reasoning with tools, self-reflection, and complex analysis
  - **OFF**: Uses a simpler conversational approach for faster responses

<div align="center">
  <img src="./screenshots/reasoning_toggle.png" alt="Reasoning Toggle" width="400"/>
  <p><em>Reasoning toggle in the sidebar</em></p>
</div>

### Model Selection

- **General RAG (llama3:8b-64k)**: Better for general questions and analysis
- **IOC Extraction (qwen2.5:7b-64k)**: Specialized for extracting and analyzing Indicators of Compromise

<div align="center">
  <img src="./screenshots/model_selection.png" alt="Model Selection" width="400"/>
  <p><em>Model selection options in the sidebar</em></p>
</div>

### Domain Role Selection

- **CTI Analyst**: Standard cyber threat intelligence analysis
- **Senior Security Analyst**: Advanced analysis with expertise in Sysmon, eBPF, Azure Entra ID, etc.

<div align="center">
  <img src="./screenshots/domain_roles.png" alt="Domain Roles" width="400"/>
  <p><em>Domain role selection in the sidebar</em></p>
</div>

### Website Processing

Process any website to make its content available for questioning:

1. Enter a URL in the "Process Website" section
2. Click "Process Website"
3. Wait for processing to complete
4. Ask questions about the website content

<div align="center">
  <img src="./screenshots/website_processing.png" alt="Website Processing" width="800"/>
  <p><em>Processing websites for enhanced intelligence</em></p>
</div>

### Web Search

Enable web search to augment responses with real-time information from the internet:

1. Toggle "Enable Web Search" in the sidebar
2. Ask questions that might benefit from current information
3. The system will automatically search the web and incorporate relevant results

<div align="center">
  <img src="./screenshots/web_search.png" alt="Web Search" width="400"/>
  <p><em>Web search toggle in the sidebar</em></p>
</div>

### Database Prioritization

After processing a website or enabling web search, you can adjust how much the system prioritizes temporary database results:

- Use the "Temporary DB Boost" slider in the sidebar
- Higher values (2.0-5.0) give more weight to processed website content and web search results
- Lower values (1.0-2.0) balance between permanent and temporary knowledge

<div align="center">
  <img src="./screenshots/db_prioritization.png" alt="Database Prioritization" width="400"/>
  <p><em>Temporary DB Boost slider in the sidebar</em></p>
</div>

## 🔄 Website Processing Workflow

When you process a website through the UI:

1. **Content Extraction**: The application scrapes the website content
2. **Text Processing**: Content is cleaned and chunked into manageable pieces
3. **Vector Embedding**: Chunks are converted to vector representations
4. **Temporary Storage**: Embeddings are stored in a temporary vector database
5. **Prioritized Retrieval**: When you ask questions, the system prioritizes content from processed websites based on the TEMP_DB_BOOST setting

<div align="center">
  <img src="./screenshots/website_workflow.png" alt="Website Processing Workflow" width="800"/>
  <p><em>Website processing workflow diagram</em></p>
</div>

## 🛠️ Troubleshooting

### Ollama Issues

If you encounter issues with Ollama, you can restart the service:

```bash
# Use the provided script
chmod +x restart_ollama.sh
./restart_ollama.sh
```

<div align="center">
  <img src="./screenshots/restart_ollama.png" alt="Restart Ollama" width="700"/>
  <p><em>Restarting the Ollama service</em></p>
</div>

### Memory Usage

If you experience high memory usage:

1. Close other memory-intensive applications
2. Reduce the number of processed websites in a single session
3. Use the IOC Extraction model which requires less memory

<div align="center">
  <img src="./screenshots/memory_usage.png" alt="Memory Usage" width="700"/>
  <p><em>Monitoring memory usage during operation</em></p>
</div>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <img src="./screenshots/oracle_footer.png" alt="Oracle Footer" width="200"/>
  <p>Developed by Dexterlabs | Author: Dhruv Majumdar</p>
</div>

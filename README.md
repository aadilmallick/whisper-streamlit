# Whisper Streamlit Audio Transcription App

A powerful web application that transcribes audio files using OpenAI's Whisper model and provides AI-powered summarization using local LLM models through Ollama. Built with Streamlit for an intuitive user interface.

## Overview

This application combines speech-to-text capabilities with natural language processing to provide:
- **Audio Transcription**: Upload audio files and get accurate text transcriptions using OpenAI's Whisper model
- **AI Summarization**: Generate concise summaries of transcribed content using Llama 3.2 via Ollama
- **Interactive Web Interface**: User-friendly Streamlit interface for seamless interaction
- **Multiple Audio Format Support**: Handles MP3, WAV, M4A, WebM, OGG, and FLAC files

## Features

- 🎤 **Multi-format Audio Support**: Upload audio in various formats (mp3, wav, m4a, webm, ogg, flac)
- 🤖 **AI-Powered Transcription**: Uses OpenAI's Whisper "base.en" model for accurate speech recognition
- 📝 **Smart Summarization**: Leverages Llama 3.2 to create concise summaries of transcriptions
- 🚀 **Optimized Performance**: Implements model caching and session state management for efficiency
- 🎯 **Clean UI**: Intuitive Streamlit interface with real-time feedback and spinners
- 🧰 **Extensible Architecture**: Includes utilities for RAG, vector stores, and MCP integration

## Technical Details

### Application Flow

1. **Initialization**
   - Loads Whisper model (base.en) into session state with caching
   - Initializes Llama 3.2 chat model via Ollama with caching
   - Models are loaded only once per session for optimal performance

2. **Audio Upload**
   - User uploads an audio file through Streamlit's file uploader
   - File is saved to a temporary location using Python's `tempfile`
   - Audio player displays the uploaded file for preview

3. **Transcription Process**
   - User clicks "Transcribe" button
   - Whisper model processes the audio file
   - Transcribed text is stored in session state
   - Temporary file is cleaned up after processing
   - Transcription is displayed in a formatted markdown blockquote

4. **Summarization Process** (Optional)
   - User can click "Summarize" to generate a summary
   - Transcribed text is sent to Llama 3.2 model
   - AI-generated summary is displayed below the transcription
   - Summary persists in session state

5. **State Management**
   - Uses Streamlit's session state to maintain:
     - Loaded models (whisper_model, chat_model)
     - Current transcription
     - Generated summary
     - Uploaded file path
   - Automatically clears previous results when new files are uploaded

### Architecture

The application follows a modular architecture:

```
┌─────────────────────────────────────┐
│   Streamlit Web Interface           │
│   (whisper_streamlit.py)            │
└─────────────┬───────────────────────┘
              │
     ┌────────┴────────┐
     │                 │
┌────▼────┐      ┌────▼─────┐
│ Whisper │      │ Langchain│
│  Model  │      │   Llama  │
└─────────┘      └──────────┘
                      │
              ┌───────┴────────┐
              │                │
        ┌─────▼─────┐    ┌────▼────┐
        │   RAG     │    │   MCP   │
        │ Utilities │    │  Client │
        └───────────┘    └─────────┘
```

### Technology Stack

- **Frontend Framework**: Streamlit 1.46.1+
- **Speech Recognition**: OpenAI Whisper (base.en model)
- **Language Model**: Llama 3.2 via Ollama (through Langchain)
- **ML Framework**: Langchain 0.3.26+
- **Vector Stores**: FAISS, ChromaDB
- **Embeddings**: Ollama Embeddings (mxbai-embed-large)
- **Document Processing**: PyPDF, YouTube Transcript API
- **Model Context Protocol**: MCP 1.10.1+

## Getting Started

### Prerequisites

- Python 3.13 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Llama 3.2 model pulled in Ollama: `ollama pull llama3.2`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aadilmallick/whisper-streamlit.git
   cd whisper-streamlit
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   # Or install from pyproject.toml
   pip install -e .
   ```

3. **Ensure Ollama is running**
   ```bash
   # Start Ollama service
   ollama serve
   
   # In another terminal, pull the required model
   ollama pull llama3.2
   ```

### Usage

1. **Run the main application**
   ```bash
   streamlit run whisper_streamlit.py
   ```

2. **Access the application**
   - Open your browser to `http://localhost:8501`

3. **Transcribe audio**
   - Upload an audio file (mp3, wav, m4a, webm, ogg, or flac)
   - Click "Transcribe" and wait for the model to process
   - View the transcription result

4. **Generate summary** (Optional)
   - After transcription completes, click "Summarize"
   - Wait for the AI to generate a concise summary
   - View the summary below the transcription

### Running the Demo Application

To explore Streamlit components and utilities:
```bash
streamlit run main.py
```

### Running the MCP Server

To test the Model Context Protocol server:
```bash
python mcp_practice.py
```

## File Structure

### Main Application Files

- **`whisper_streamlit.py`** - Main Streamlit application for audio transcription and summarization
  - Handles file upload and audio playback
  - Manages Whisper model loading and transcription
  - Integrates Llama 3.2 for text summarization
  - Implements session state management for models and results

- **`main.py`** - Streamlit demo application showcasing various UI components
  - Examples of forms, buttons, tabs, columns, and containers
  - Demonstrates session state management patterns
  - Includes custom `STState` and `STForm` classes for state and form management
  - Useful reference for Streamlit development patterns

### Utility Modules

- **`utils/langchain_llms.py`** - Langchain LLM wrapper
  - `LangchainLLama` class for interacting with Ollama models
  - Provides simple interface for getting responses from local LLMs
  - Supports tool usage for function calling capabilities

- **`utils/langchain_rag.py`** - RAG (Retrieval Augmented Generation) utilities
  - `LangchainRAG`: Core RAG functionality for similarity search
  - `FAISSVectorStore`: FAISS vector store operations (create, save, load)
  - `ChromaVectorStore`: ChromaDB operations with deduplication
  - `DocumentLoaders`: Load and split documents from multiple sources:
    - YouTube videos (via transcript API)
    - PDF files
    - Text files and directories
    - Web pages
  - Implements document chunking and embedding generation

- **`mcpUtils.py`** - Model Context Protocol (MCP) client implementation
  - `MCPClient` class for connecting to MCP servers
  - Supports both stdio and SSE (Server-Sent Events) transport
  - Provides async interface for tool discovery and execution
  - Enables integration with MCP-compatible services

### Example/Practice Files

- **`mcp_practice.py`** - Simple MCP server example
  - Demonstrates FastMCP server setup
  - Implements basic arithmetic tools (add, subtract)
  - Shows stdio transport configuration
  - Useful for testing MCP client functionality

### Configuration Files

- **`pyproject.toml`** - Project configuration and dependencies
  - Project metadata (name: pythonai, version: 0.1.0)
  - Python version requirement (>=3.13)
  - Complete dependency list including:
    - ML/AI: whisper, langchain, ollama integrations
    - Vector stores: chromadb, faiss-cpu
    - UI: streamlit
    - Document processing: pypdf, youtube-transcript-api

- **`Dockerfile`** - Docker containerization configuration
  - Sets up containerized environment for deployment
  - Includes all necessary dependencies

- **`.gitignore`** - Git ignore patterns
  - Excludes virtual environments, cache files, and build artifacts

- **`.python-version`** - Python version specification for pyenv

### Notebook Files

- **`main.ipynb`** - Jupyter notebook for experimentation
- **`whipser.ipynb`** - Jupyter notebook for Whisper model exploration

## Advanced Features

### RAG Capabilities

The application includes comprehensive RAG utilities for building knowledge bases:

```python
from utils.langchain_rag import DocumentLoaders, FAISSVectorStore, LangchainRAG

# Load documents from various sources
loader = DocumentLoaders()
docs = loader.get_docs_from_pdf("./data/")

# Create vector store
vector_store = FAISSVectorStore()
db = vector_store.create_FAISS_from_docs(docs)

# Query similar documents
rag = LangchainRAG()
related_docs = rag.get_related_docs_from_query(db, "your query here")
```

### MCP Integration

Connect to Model Context Protocol servers for extended functionality:

```python
from mcpUtils import MCPClient
from mcp import StdioServerParameters

client = MCPClient(server_init=("stdio", StdioServerParameters(
    command="python",
    args=["mcp_practice.py"],
)))

async def callback(session):
    tools = await client.get_tools(session)
    # Use MCP tools

client.on_connection_established(callback)
await client.connect_to_mcp_server()
```

## Dependencies

Key dependencies include:
- `streamlit` - Web application framework
- `openai-whisper` - Speech recognition
- `langchain` & `langchain-ollama` - LLM integration
- `faiss-cpu` & `chromadb` - Vector databases
- `pypdf` - PDF processing
- `youtube-transcript-api` - YouTube transcript extraction
- `mcp` - Model Context Protocol support

For a complete list, see `pyproject.toml`.

## Troubleshooting

### Common Issues

1. **Ollama connection errors**
   - Ensure Ollama is running: `ollama serve`
   - Verify Llama 3.2 is installed: `ollama list`

2. **Whisper model loading errors**
   - First run downloads the model (~140MB)
   - Ensure stable internet connection
   - Check disk space for model storage

3. **Memory issues**
   - Whisper base.en model requires ~1GB RAM
   - Llama 3.2 requires additional ~2-4GB RAM
   - Consider using smaller models or adding swap space

4. **Audio file format errors**
   - Ensure ffmpeg is installed for format conversion
   - Try converting audio to WAV format first

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI for the Whisper model
- Meta for Llama models
- Ollama team for local LLM infrastructure
- Streamlit team for the excellent web framework
- Langchain team for the LLM orchestration framework

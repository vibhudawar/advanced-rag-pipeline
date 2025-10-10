# 🤖 Production-Ready RAG Application

A complete Retrieval-Augmented Generation (RAG) application with advanced features including query expansion, reranking, and conversation memory. Built with LangChain, LangGraph, and Streamlit.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=flat&logo=pinecone&logoColor=white)](https://pinecone.io)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Deployment to Streamlit Cloud](#-deployment-to-streamlit-cloud)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### Document Ingestion
- 📄 **Multi-format Support**: PDF, DOCX, TXT, Markdown
- 🔪 **Advanced Chunking Strategies**:
  - **Recursive**: Fast, fixed-size chunks with configurable size and overlap
  - **Semantic**: Context-aware splitting using embeddings
  - **Agentic**: AI-powered intelligent proposition extraction
- 🗄️ **Vector Storage**: Pinecone integration with automatic index management
- 📊 **Progress Tracking**: Real-time logs and processing statistics

### RAG Query Pipeline
- 🔍 **Query Expansion**: Multi-query retrieval for better results
- 🎯 **Smart Reranking**: Cohere API or HuggingFace cross-encoders
- 🤖 **Query Classification**: Intelligent routing (greeting vs RAG query)
- 💬 **Conversation Memory**: Persistent chat history with LangGraph
- 📚 **Source Citations**: View retrieved document chunks with relevance scores
- 🎨 **Score Filtering**: Automatic filtering of low-quality sources (< 0.1 threshold)

### User Interface
- 🎯 **Clean Design**: Modern Streamlit interface
- 📑 **Two-Tab Layout**: Separate views for ingestion and chat
- 💾 **Session Management**: Persistent conversations
- 🔄 **Real-time Updates**: Live processing logs
- 📱 **Responsive**: Works on desktop and mobile

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │  Document Ingestion Tab  │  │      Chat Tab            │   │
│  │  - Upload files          │  │  - Query documents       │   │
│  │  - Configure chunking    │  │  - View sources          │   │
│  │  - View statistics       │  │  - Chat history          │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (main_app.py)                      │
│                                                                 │
│  ┌────────────────────┐         ┌──────────────────────────┐  │
│  │ Ingestion Pipeline │         │    RAG Pipeline          │  │
│  │                    │         │    (LangGraph)           │  │
│  │ 1. Parse docs      │         │                          │  │
│  │ 2. Chunk text      │         │  Classify → Expand →     │  │
│  │ 3. Embed chunks    │         │  Retrieve → Rerank →     │  │
│  │ 4. Store in DB     │         │  Generate                │  │
│  └────────────────────┘         └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                            │
│                                                                 │
│  📊 Pinecone VectorDB  |  🤖 OpenAI/Gemini  |  🎯 Cohere      │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow (LangGraph)

```
START → Classify Query
          ↓
    ┌─────┴─────┐
    ↓           ↓
Greeting?    RAG Query?
    ↓           ↓
Simple      Expand Query
Response        ↓
    ↓       Retrieve Docs (Multi-query)
    ↓           ↓
    ↓       Rerank (Score > 0.1)
    ↓           ↓
    └────→  Generate Answer
                ↓
              END
```

---

## 📂 Project Structure

```
RAG/
├── app.py                          # Entry point for Streamlit
├── config.py                       # Configuration (loads .env)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── src/
│   ├── main_app.py                # 🔴 BACKEND: Core RAG pipeline
│   ├── rag_frontend.py            # 🎨 FRONTEND: Streamlit UI
│   │
│   ├── ingestion/                 # Document processing
│   │   ├── __init__.py
│   │   ├── DocumentParsers.py    # PDF, DOCX, TXT, MD parsers
│   │   ├── ChunkCreator.py       # Recursive, Semantic, Agentic chunking
│   │   ├── EmbeddingCreator.py   # OpenAI/HuggingFace embeddings
│   │   └── DBIngestion.py        # Pinecone vector store operations
│   │
│   ├── generation/                # LLM generation
│   │   ├── __init__.py
│   │   └── llm_generator.py      # OpenAI/Gemini + query expansion
│   │
│   ├── reranking/                 # Document reranking
│   │   ├── __init__.py
│   │   └── reranker.py           # Cohere + Cross-encoder rerankers
│   │
│   ├── retrieval/                 # (Future: advanced retrieval)
│   │   └── __init__.py
│   │
│   ├── utils/                     # Helper functions
│   │   ├── __init__.py
│   │   ├── Helpers.py            # Utility functions
│   │   ├── PromptsConstants.py  # System prompts
│   │   └── AgenticChunkerHelper.py  # Agentic chunking logic
│   │
│   └── ui/                        # UI components
│       ├── __init__.py
│       └── components.py         # Reusable Streamlit components
│
└── .env                           # Environment variables (create this)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Step 1: Clone or Download

```bash
cd /path/to/RAG
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n rag-app python=3.9
conda activate rag-app
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Create `.env` File

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws  # Your Pinecone environment

# Optional
COHERE_API_KEY=your-cohere-api-key        # For reranking (recommended)
GEMINI_API_KEY=your-gemini-api-key        # Alternative LLM provider
LANGCHAIN_API_KEY=your-langsmith-key      # For tracing/debugging
LANGCHAIN_TRACING_V2=true                 # Enable LangSmith tracing

# Provider Selection (defaults to OpenAI)
LLM_PROVIDER=openai                       # or "gemini"
EMBEDDING_PROVIDER=openai                 # or "huggingface"
```

### Environment Variables Explained

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for embeddings and LLM |
| `PINECONE_API_KEY` | ✅ Yes | Pinecone API key for vector storage |
| `PINECONE_ENVIRONMENT` | ✅ Yes | Your Pinecone environment region |
| `COHERE_API_KEY` | ⚠️ Recommended | Cohere API for better reranking |
| `GEMINI_API_KEY` | ❌ Optional | Use Google's Gemini instead of OpenAI |
| `LANGCHAIN_API_KEY` | ❌ Optional | Enable LangSmith tracing for debugging |

---

## 🎮 Running the Application

### Quick Start (Recommended)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Alternative Methods

```bash
# Method 1: From root directory
python -m streamlit run src/rag_frontend.py

# Method 2: Direct run
cd src
streamlit run rag_frontend.py
```

---

## 📖 Usage Guide

### 1️⃣ Document Ingestion

1. **Navigate to "Document Ingestion" tab** (default view)

2. **Configure Chunking Strategy**:
   - **Recursive**: Shows chunk size and overlap sliders (good for most cases)
   - **Semantic**: No configuration needed (context-aware, slower)
   - **Agentic**: No configuration needed (AI-powered, slowest but best quality)

3. **Upload Documents**:
   - Click "Browse files" or drag & drop
   - Supported formats: PDF, DOCX, TXT, MD
   - Multiple files supported

4. **Process Documents**:
   - Click "Process Documents"
   - Watch real-time logs in the expander
   - View results table with statistics

5. **View Statistics**:
   - Total files processed
   - Total chunks created
   - Success rate
   - Individual file results

### 2️⃣ Chat with Documents

1. **Navigate to "Chat" tab**

2. **Ask Questions**:
   - Type your question in the chat input
   - Press Enter or click send
   - Wait for AI response (streaming)

3. **View Sources**:
   - Click "View Sources" below each response
   - Expand individual sources to see document chunks
   - Check relevance scores

4. **Manage Conversations**:
   - **New Chat**: Start fresh conversation (sidebar)
   - **Select Thread**: Resume previous conversation
   - **Delete Thread**: Remove old conversations

### 💡 Pro Tips

- **Query Classification**: The system automatically detects if your query needs document retrieval or is just a greeting/simple question
- **Source Filtering**: Only sources with score ≥ 0.1 are shown (automatic quality filtering)
- **Conversation Context**: Each thread maintains its own conversation history
- **Score Interpretation**: Higher scores (closer to 1.0) indicate more relevant sources

---

## ☁️ Deployment to Streamlit Cloud

### Pre-Deployment Checklist

✅ All code is cloud-ready (no filesystem dependencies)
✅ Pinecone stores all document data (persists across restarts)
✅ Conversation history in SQLite (ephemeral but acceptable)

### Step 1: Prepare Repository

```bash
# Ensure requirements.txt is up to date
pip freeze > requirements.txt

# Create .streamlit/config.toml (optional)
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
EOF
```

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: RAG application"
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set **Main file path**: `app.py`
5. Set **Python version**: 3.9 or higher

### Step 4: Configure Secrets

In Streamlit Cloud dashboard, go to **App settings → Secrets** and add:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENVIRONMENT = "us-east-1-aws"
COHERE_API_KEY = "your-cohere-api-key"
LANGCHAIN_API_KEY = "your-langsmith-key"
LANGCHAIN_TRACING_V2 = "true"
```

### Step 5: Deploy!

Click "Deploy" and wait for the app to build (2-3 minutes).

### ⚠️ Important Notes for Cloud Deployment

- **Conversation History**: Will be lost on app restart (ephemeral SQLite)
  - For production, consider using PostgreSQL via Streamlit's connection feature
- **Document Metadata**: Stored in Pinecone (persists across restarts) ✅
- **File Uploads**: Processed immediately, not stored locally ✅

---

## 🎯 Advanced Features

### Query Expansion

The system automatically generates multiple query variants to improve retrieval:

```python
Original Query: "What is the loan interest rate?"

Expanded Queries:
1. "What is the loan interest rate?"
2. "What is the APR for the loan?"
3. "What are the loan terms and interest charges?"
```

### Reranking with Score Filtering

Documents are reranked using Cohere's `rerank-v3.5` model and filtered:

```python
Retrieved: 10 documents
Reranked:  10 documents
Filtered:  5 documents (score >= 0.1)
→ Only high-quality sources shown to user
```

### Query Classification

Intelligent routing saves tokens and improves UX:

```
User: "Hi"
→ Classification: greeting
→ Simple response (no RAG)

User: "What is the refund policy?"
→ Classification: rag_query
→ Full RAG pipeline with document retrieval
```

### Agentic Chunking

AI-powered chunking extracts atomic propositions:

```
Original Text:
"John works at Microsoft. Microsoft is located in Seattle.
Seattle is in Washington state."

Propositions:
1. John works at Microsoft
2. Microsoft is located in Seattle
3. Seattle is in Washington state

→ Each proposition becomes a semantically complete chunk
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'src'`

**Solution**: Always run from the root directory using `app.py`:
```bash
streamlit run app.py
```

#### 2. API Key Errors

**Symptoms**:
```
The api_key client option must be set...
```

**Solution**:
1. Check `.env` file exists in project root
2. Verify API keys are correct
3. Restart the application

#### 3. Pinecone Index Not Found

**Symptoms**:
```
Index 'rag-documents' does not exist
```

**Solution**: The index is created automatically on first document upload. Just upload documents via the Ingestion tab.

#### 4. LangChain Deprecation Warnings

**Symptoms**:
```
LangChainDeprecationWarning: ...
```

**Solution**: These are warnings, not errors. The code uses the latest LangChain APIs. Upgrade LangChain if needed:
```bash
pip install --upgrade langchain langchain-core langchain-openai
```

#### 5. Streamlit Duplicate Key Errors

**Fixed**: The app now generates unique message IDs to prevent key conflicts.

#### 6. Conversation History Not Loading

**Symptoms**: Previous conversations don't load

**Solution**:
- Check `chatbot.db` exists in project root
- On Streamlit Cloud, conversations reset on restart (this is expected)
- For persistent history, upgrade to PostgreSQL

### Debug Mode

Enable LangSmith tracing for detailed debugging:

```bash
# In .env
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

Then view traces at [smith.langchain.com](https://smith.langchain.com)

---

## 📊 Performance Tips

### For Faster Ingestion
- Use **Recursive chunking** (fastest)
- Upload files in batches of 5-10
- Optimize chunk size: 500-1000 characters works well

### For Better Retrieval
- Use **Semantic or Agentic chunking** (slower but better quality)
- Enable **Cohere reranking** (significant improvement)
- Set `top_k=10` for retrieval, `rerank_top_k=5` for final results

### For Lower Costs
- Use `gpt-3.5-turbo` instead of `gpt-4` (edit `config.py`)
- Reduce `top_k` to retrieve fewer documents
- Use Gemini as alternative (cheaper)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for more document formats (HTML, CSV, Excel)
- [ ] Implement hybrid search (keyword + vector)
- [ ] Add document deletion feature
- [ ] Support for multiple Pinecone indexes
- [ ] Add authentication/user management
- [ ] Implement streaming for reranking
- [ ] Add evaluation metrics (RAGAS, etc.)

---

## 📝 License

This project is provided as-is for educational and commercial use.

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Graph-based workflows
- [Streamlit](https://streamlit.io) - Web UI framework
- [Pinecone](https://pinecone.io) - Vector database
- [OpenAI](https://openai.com) - LLM and embeddings
- [Cohere](https://cohere.com) - Reranking

---

## 📧 Support

For issues and questions:
- Check the [Troubleshooting](#-troubleshooting) section
- Review LangChain docs: [python.langchain.com](https://python.langchain.com)
- Review Streamlit docs: [docs.streamlit.io](https://docs.streamlit.io)

---

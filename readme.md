# 📰 GenAI Newspaper Article Generator

A production-ready RAG (Retrieval-Augmented Generation) system for generating professional newspaper articles using LangChain, OpenAI, FAISS vector store, and Streamlit.

## 🏗️ Architecture

```
┌─────────────────────┐
│   Streamlit UI      │  ← User Interface
└──────────┬──────────┘
           │
           ↓
    ┌──────────────┐
    │ Input Mode:  │
    │ • Wikipedia  │
    │ • Upload Doc │
    └──────┬───────┘
           │
           ↓
┌──────────────────────┐
│   Research Agent     │  ← Wikipedia/Document Retrieval
│   (Optional)         │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│   Text Chunking      │  ← RecursiveCharacterTextSplitter
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│   FAISS Vector Store │  ← OpenAI Embeddings
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│   Writer Agent       │  ← GPT-4 Article Generation
│   (Optional)         │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│   Newspaper Article  │  ← 3 Paragraphs + Headline
└──────────────────────┘
```

## 🚀 Features

- ✅ **Dual Input Modes**: Wikipedia search OR document upload (PDF/TXT)
- ✅ **RAG Pipeline**: FAISS vector store with OpenAI embeddings
- ✅ **Two-Agent System**: Optional research + writer agent orchestration
- ✅ **Professional Output**: Newspaper-style formatting with headline
- ✅ **Docker Support**: Full containerization with docker-compose
- ✅ **Persistent Storage**: FAISS index persistence between sessions
- ✅ **Download Articles**: Export generated content as TXT

## 📦 Project Structure

```
mb-genai-newspaper/
├── app.py                  # Streamlit UI
├── rag.py                  # Core RAG pipeline
├── agents.py               # Two-agent orchestration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── .env.example            # Environment variables template
├── README.md               # This file


## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **UI Framework** | Streamlit |
| **RAG Framework** | LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | OpenAI text-embedding-3-small |
| **LLM** | OpenAI GPT-4o-mini |
| **Data Sources** | Wikipedia API + PDF/TXT Upload |
| **Containerization** | Docker & Docker Compose |

## 📋 Prerequisites

- **Docker & Docker Compose** (recommended)
- OR **Python 3.11+** (for local development)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

## 🚀 Quick Start (Docker)

### 1. Clone and Setup

```bash
# Clone or create project directory
mkdir mb-genai-newspaper && cd mb-genai-newspaper

# Create all required files (app.py, rag.py, agents.py, etc.)
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

**Required in .env:**
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Build and Run

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

### 4. Access Application

Open your browser: **http://localhost:8501**

## 🖥️ Local Development (Without Docker)

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 4. Run Application

```bash
streamlit run app.py
```

## 📖 Usage Guide

### Wikipedia Mode

1. Select **"Wikipedia search"**
2. Enter a topic (e.g., "Artificial Intelligence")
3. Adjust max pages slider (1-10)
4. Click **"Generate Article"**

### Document Upload Mode

1. Select **"Upload documents"**
2. Upload one or more PDF/TXT files
3. Enter the article topic
4. Click **"Generate Article"**

### Two-Agent System

1. Enable **"Use Two-Agent System"** in sidebar
2. Only works with Wikipedia mode
3. Demonstrates agent orchestration:
   - **Research Agent**: Gathers information
   - **Writer Agent**: Generates article

## 🎯 How It Works

### Standard RAG Pipeline

1. **Input**: User provides topic and selects data source
2. **Retrieval**: 
   - Wikipedia: Searches and fetches article content
   - Upload: Extracts text from PDF/TXT files
3. **Chunking**: Text split into 1000-character chunks
4. **Vectorization**: Chunks embedded using OpenAI embeddings
5. **Storage**: Embeddings stored in FAISS vector database
6. **Retrieval**: Most relevant chunks retrieved (k=4)
7. **Generation**: GPT-4 generates article with context
8. **Output**: Professional newspaper article

### Two-Agent System

```python
# Research Agent
context = research_agent(topic)

# Writer Agent  
article = writer_agent(context, topic)

# Orchestration
final_article = two_agent_pipeline(topic)
```

## 🐳 Docker Commands

### Basic Operations

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after code changes
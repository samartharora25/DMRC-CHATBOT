# HR Document RAG Chatbot

A sophisticated HR Document RAG (Retrieval-Augmented Generation) chatbot system built using local LLMs, vector embeddings, and advanced document processing capabilities. This system processes HR policy documents, creates structured chunks, generates embeddings, and provides an interactive conversational interface for HR-related queries.

## 🚀 Features

- **Advanced PDF Processing**: Intelligent chunking of HR documents with chapter and subtopic detection
- **Local Model Support**: Uses locally hosted Llama 3.2 3B and BGE embeddings models
- **Vector Database**: ChromaDB for efficient document retrieval
- **Interactive RAG System**: Real-time question-answering with document context
- **Document Summarization**: Automated policy summarization with privacy redaction
- **Q&A Dataset Generation**: Automated creation of training datasets from documents
- **Multi-API Support**: Integration with Groq, Together AI, and local models

## 📋 Prerequisites

- **Python**: 3.11.11 
- **CUDA**: GPU with CUDA support (recommended for model inference)
- **Memory**: Minimum 16GB RAM, 6GB VRAM minimum
- **Storage**: ~15GB for models and data

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Shar-mayank0/DMRC_Chatbot.git
cd DMRC_Chatbot/chatbot
```

### 2. Install Dependencies
This project uses `uv` for dependency management:

```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

### 3. Environment Setup
Create a `.env` file in the project root:

```bash
cp .env.sample .env
```

Edit the `.env` file with your API keys:
```bash
HF_ACCESS_KEY="your_huggingface_token_here"
TOGETHER_API_KEY="your_together_ai_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

**Required API Keys:**
- **HF_ACCESS_KEY**: Hugging Face token for model downloads
- **GROQ_API_KEY**: For fast LLM inference (used in interactive sessions)
- **TOGETHER_API_KEY**: For Q&A dataset generation (optional)

### 4. Download Models
Run the model downloader to fetch required models locally:

```bash
uv run model_downloader.py
```

This will download:
- **Llama-3.2-3B-Instruct**: Local language model (~6GB)
- **BGE-large-en-v1.5**: Embedding model (~1.3GB)

**Important Notes:**
- **Custom Models**: To download different models, edit `model_downloader.py` and modify these lines:
  ```python
  model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Change this to your desired model
  local_dir = os.path.join(os.getcwd(), "models/meta-llama/Llama-3.2-3B-Instruct")  # Update path accordingly
  ```
- **Open Source Only**: This downloader only supports open-source models available on Hugging Face
- **Model Compatibility**: Ensure the model is compatible with the transformers library and your hardware specifications

Models are saved in the `models/` directory for offline use.

## 📁 Project Structure

```
chatbot/
├── src/                           # Core modules
│   ├── chunker.py                # PDF processing and chunking
│   ├── embedder.py               # Document embedding generation
│   ├── langchain_rag.py          # RAG system implementation
│   └── summarizer.py             # Document summarization
├── data/                         # Input documents
│   └── Sample HR Policy Manual.pdf
├── models/                       # Downloaded local models
│   ├── BAAI/bge-large-en-v1.5/   # Embedding model
│   └── meta-llama/Llama-3.2-3B-Instruct/  # Language model
├── hr_chroma_db/                 # Vector database
├── main.py                       # Main processing pipeline
├── interactive_rag_session.py    # Interactive chat interface
├── model_downloader.py           # Model download utility
├── QAagent.py                    # Q&A dataset generation
├── test.py                       # System testing utilities
├── pyproject.toml                # Project dependencies
└── .env.sample                   # Environment template
```

## 📖 Core Components

### 1. Document Chunker (`src/chunker.py`)

**Important Note**: This chunker is specifically designed for the sample HR policy document format. For different documents, you'll need to modify the chunker parameters.

**Key Classes:**
- `SubtopicBlock`: Represents individual subtopic chunks
- `ChapterBlock`: Represents complete chapter chunks  
- `PDFChunker`: Main processing class

**Key Methods:**
```python
def parse_toc_structure(start_page=4, end_page=10, offset=10, last_page=208)
def extract_subtopic_and_chapter_chunks(toc)
def is_subtopic_heading(text, font_name, color_val)
def save_chunks(subtopics, chapters, filename)
```

**Customization for Different Documents:**
When using your own HR documents, modify these parameters in `chunker.py`:
- `HEADING_FONT`: Font used for headings
- `HEADING_COLOR`: Color value for headings
- `parse_toc_structure()` parameters for TOC location
- `is_subtopic_heading()` criteria for your document format

### 2. Embedding Generator (`src/embedder.py`)

Generates vector embeddings using the local BGE model:
- **Model**: BAAI/bge-large-en-v1.5
- **Device**: CUDA if available, CPU fallback
- **Output**: 1024-dimensional embeddings

### 3. RAG System (`src/langchain_rag.py`)

Complete RAG implementation with:
- **Vector Store**: ChromaDB for document storage and retrieval
- **LLM Integration**: Groq API for fast inference
- **Custom Embeddings**: Local BGE model wrapper
- **Retrieval**: Similarity search with metadata filtering

### 4. Document Summarizer (`src/summarizer.py`)

Features:
- **Local Model**: Uses Llama-3.2-3B-Instruct
- **Privacy Protection**: Automatic PII redaction
- **Batch Processing**: Handles multiple documents
- **Progress Tracking**: Resume interrupted processes

### 5. Q&A Agent (`QAagent.py`)

Automated Q&A dataset generation:
- **LangGraph**: State-based processing workflow
- **API Integration**: Together AI for dataset generation
- **Quality Control**: Multiple review iterations
- **Output**: JSONL format for training

## 🚀 Usage

### 1. Complete Pipeline Execution

Run the full processing pipeline:

```bash
uv run main.py
```

This will:
1. ✅ Process the PDF document into chunks
2. ✅ Generate embeddings for all chunks  
3. ✅ Build ChromaDB vector store
4. ✅ Summarize all documents
5. ✅ Save progress for resumability

### 2. Interactive Chat Session

Start the interactive RAG chatbot:

```bash
uv run interactive_rag_session.py
```

**Available Commands:**
- Ask any HR policy question
- `stats` - View system statistics
- `debug <query>` - Debug search results
- `rebuild` - Rebuild vector store
- `test` - Run system tests
- `quit/exit` - Exit session

**Example Queries:**
```
🤔 Your question: What is the employee onboarding process?
🤔 Your question: How are performance evaluations conducted?
🤔 Your question: What are the leave policies?
```

### 3. Generate Q&A Dataset

Create training datasets from your documents:

```bash
uv run QAagent.py
```

Output: `qa_dataset.jsonl` with structured Q&A pairs

### 4. System Testing

Verify your installation:

```bash
uv run test.py
```

## ⚙️ Configuration

### Model Configuration

**For different memory constraints:**

Edit `model_downloader.py` to adjust memory allocation:
```python
max_memory = {
    0: "6GiB",      # GPU memory
    "cpu": "6GiB"   # CPU memory
}
```

### RAG Configuration

Edit `interactive_rag_session.py` for different models:
```python
rag.build_rag_system(
    force_rebuild=False,
    groq_model="llama3-8b-8192",  # or "llama3-70b-8192"
    temperature=0.1,
    max_tokens=1024
)
```

### Document Processing

For different document formats, modify `chunker.py`:
```python
# Update these for your document structure
HEADING_FONT = "Your-Document-Font"
HEADING_COLOR = your_color_value

# Adjust TOC parsing parameters
def parse_toc_structure(start_page=X, end_page=Y, offset=Z, last_page=N)
```

## 📊 Output Files

| File | Description | Usage |
|------|-------------|--------|
| `chunks.json` | Processed document chunks | Input for RAG system |
| `hr_chroma_db/` | Vector database | Document retrieval |
| `summaries.json` | Document summaries | Analysis and review |
| `qa_dataset.jsonl` | Q&A pairs | Model training |
| `summarization_progress.json` | Processing progress | Resume interrupted tasks |

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce model memory allocation in model_downloader.py
max_memory = {"0": "4GiB", "cpu": "4GiB"}
```

**2. API Key Errors**
```bash
# Verify .env file contains valid keys
cat .env
```

**3. Missing Models**
```bash
# Re-download models
uv run model_downloader.py
```

**4. ChromaDB Issues**
```bash
# Force rebuild vector store
# In interactive_rag_session.py set: force_rebuild=True
```

**5. Document Processing Errors**
- Ensure PDF is in `data/` directory
- Check chunker configuration for your document format
- Verify document structure matches expected format

### Performance Optimization

**For faster processing:**
1. Use GPU acceleration (CUDA)
2. Increase batch sizes for embeddings
3. Use smaller models for testing
4. Enable persistent ChromaDB caching

**For lower memory usage:**
1. Process documents in smaller batches
2. Use CPU-only mode
3. Reduce context window size
4. Clear cache between operations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for model hosting and transformers library
- **LangChain** for RAG framework
- **ChromaDB** for vector database
- **Groq** for fast LLM inference
- **Meta** for Llama models
- **BAAI** for BGE embedding models

## 📧 Contact

**Repository**: [DMRC_Chatbot](https://github.com/Shar-mayank0/DMRC_Chatbot)
**Project Author**: 
1.Mayank Sharma 
2.Samarth Arora
3.Devang Jain
4.Hirday Singh
5.Sarthak Pandey

6.Adya Jain        



---

⭐ **Star this repository if you found it helpful!** ⭐

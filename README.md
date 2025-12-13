# RAG Documentation Service

A cross-platform, open-source RAG (Retrieval-Augmented Generation) API service for document retrieval and question answering. Built with FastAPI, FAISS/ChromaDB, and sentence-transformers.

## Features

- **100% Open Source** - No API costs, runs entirely on your hardware
- **Cross-Platform** - Works on Windows, Linux, and macOS
- **GPU Accelerated** - CUDA (Windows/Linux) and MPS (Apple Silicon) support
- **Multiple File Formats** - PDF, Markdown, TXT, XML, HTML, DOCX, and code files
- **Configurable Models** - Choose embedding models based on your available VRAM
- **Dual Vector Stores** - FAISS (fastest) or ChromaDB (simpler)
- **GraphRAG Hybrid Search** - Combines vector similarity with knowledge graphs
- **Web UI** - Drag-and-drop Gradio interface with smart port detection
- **Production Ready** - Docker support, health checks, API versioning

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/youruser/rag-documentation-service.git
cd rag-documentation-service

# Install dependencies
pip install -e .

# Or with development tools
pip install -e ".[dev]"
```

### Run the Service

```bash
# Development mode with hot reload
uvicorn rag_service.main:app --reload

# Or using the module
python -m rag_service

# Or using make
make run
```

The API is now available at http://localhost:8000

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Launch the Web UI

The service includes a drag-and-drop web interface powered by Gradio:

```bash
# Start the API first (in one terminal)
python -m rag_service

# Launch the Web UI (in another terminal)
python -m rag_service.ui.app

# Or using the CLI command
rag-ui

# With custom options
rag-ui --port 8080 --api-url http://localhost:8000 --share
```

The UI automatically finds an available port (starting from 7860) if your preferred port is in use.

**UI Features:**
- 📁 Drag-and-drop file upload
- 📂 Index entire folders
- 🔍 Semantic search with auto/vector/graph/hybrid strategies
- 📦 Collection management
- ℹ️ System status monitoring

### Basic Usage

#### 1. Index Documents

```bash
# Index a single file
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/documents/protocol.pdf", "collection": "docs"}'

# Index a directory
curl -X POST http://localhost:8000/api/v1/ingest/directory \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/documents", "collection": "docs", "recursive": true}'
```

#### 2. Query Documents

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does MAVLink authentication work?", "top_k": 5}'
```

#### 3. Python Client

```python
import httpx

# Query the RAG service
response = httpx.post(
    "http://localhost:8000/api/v1/query",
    json={"question": "What is MAVLink?", "top_k": 3}
)
results = response.json()

# Use results as context for your LLM
context = "\n\n".join(chunk["text"] for chunk in results["chunks"])
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `VECTOR_STORE_BACKEND` | `faiss` | `faiss` or `chroma` |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

### GraphRAG Configuration

Enable hybrid vector + knowledge graph search:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_GRAPH_RAG` | `false` | Enable graph-based retrieval |
| `GRAPH_STORE_BACKEND` | `memory` | `memory` (NetworkX) or `neo4j` |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `ROUTER_MODE` | `pattern` | Query routing: `pattern` or `llm` |
| `ENTITY_EXTRACTION_MODE` | `rule_based` | `rule_based` or `llm` |
| `ENTITY_EXTRACTION_DOMAIN` | `general` | `general` or `mavlink` |

**When to enable GraphRAG:**
- Your documents contain highly relational data (protocols, state machines)
- You need multi-hop reasoning ("What happens after state X?")
- You want to query entity relationships explicitly

## Choosing Your Embedding Model

Select based on your available VRAM:

### CPU / Low Memory (4GB)

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

- **Dimensions**: 384
- **Speed**: ~100 docs/sec on CPU
- **Quality**: Good for general use
- **Memory**: ~90MB

### Consumer GPU (8-16GB VRAM)

```env
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

- **Dimensions**: 1024
- **Speed**: ~500 docs/sec on GPU
- **Quality**: Great accuracy, top MTEB scores
- **Memory**: ~1.3GB

### Research Lab (24-50GB VRAM)

```env
EMBEDDING_MODEL=intfloat/e5-mistral-7b-instruct
```

- **Dimensions**: 4096
- **Speed**: ~200 docs/sec
- **Quality**: Excellent, LLM-based embeddings
- **Memory**: ~28GB

### High-End Research (50-100GB VRAM)

```env
EMBEDDING_MODEL=Salesforce/SFR-Embedding-Mistral
```

- **Dimensions**: 4096
- **Speed**: ~150 docs/sec
- **Quality**: State-of-the-art MTEB scores
- **Memory**: ~60GB

### Multi-GPU Cluster (100-200GB+ VRAM)

```env
EMBEDDING_MODEL=nvidia/NV-Embed-v2
```

- **Dimensions**: 4096
- **Quality**: Best available
- **Memory**: ~120GB (multi-GPU recommended)

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/info` | System information |
| `POST` | `/api/v1/query` | Query documents (vector only) |
| `POST` | `/api/v1/query/hybrid` | Query with smart routing (vector/graph/hybrid) |
| `GET` | `/api/v1/query/explain` | Explain how a query would be routed |
| `POST` | `/api/v1/ingest/file` | Index a file |
| `POST` | `/api/v1/ingest/directory` | Index a directory |
| `GET` | `/api/v1/collections` | List collections |
| `DELETE` | `/api/v1/collections/{name}` | Delete a collection |
| `GET` | `/api/v1/graph/stats/{collection}` | Graph statistics |

### Query Request

```json
{
  "question": "How does authentication work?",
  "top_k": 5,
  "collection": "documents",
  "strategy": "auto"
}
```

**Strategy options:**
- `auto` - Smart routing based on query type (recommended)
- `vector` - Vector similarity search only
- `graph` - Knowledge graph traversal only
- `hybrid` - Both vector and graph search

### Query Response

```json
{
  "chunks": [
    {
      "text": "Authentication in MAVLink uses...",
      "metadata": {"source": "protocol.pdf", "chunk_index": 3},
      "score": 0.89,
      "document_id": "abc123"
    }
  ],
  "graph_context": [
    {
      "entity": "MAVLink",
      "entity_type": "Protocol",
      "relationship": "USES",
      "connected_to": "HMAC-SHA256"
    }
  ],
  "sources": ["protocol.pdf"],
  "token_estimate": 450,
  "query": "How does authentication work?",
  "collection": "documents",
  "strategy_used": "hybrid",
  "router_reasoning": "Query asks about process relationships"
}
```

## Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Native + OCR support |
| Markdown | `.md`, `.markdown` | Full support |
| Text | `.txt` | All encodings |
| XML | `.xml` | Protocol definitions |
| HTML | `.html`, `.htm` | Web pages |
| Word | `.docx` | Microsoft Word |
| Code | `.py`, `.js`, `.ts`, `.java`, etc. | 20+ languages |
| Config | `.json`, `.yaml`, `.toml`, `.ini` | Configuration files |

## Platform-Specific Setup

### Windows

```powershell
# Install with pip
pip install -e .

# For GPU support (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Run
python -m rag_service
```

### Linux

```bash
# Install system dependencies for PDF support
sudo apt install libmagic1 poppler-utils

# Install Python package
pip install -e .

# Run
python -m rag_service
```

### macOS

```bash
# Install system dependencies
brew install libmagic poppler

# Install Python package
pip install -e .

# For Apple Silicon GPU (MPS)
# PyTorch automatically detects MPS

# Run
python -m rag_service
```

### OCR Support (Optional)

For scanned PDFs:

**Windows:**
```powershell
winget install UB-Mannheim.TesseractOCR
# Add C:\Program Files\Tesseract-OCR to PATH
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

Then install Python OCR dependencies:
```bash
pip install -e ".[ocr]"
```

## Docker

### Build and Run

```bash
# Build image
docker build -t rag-service .

# Run container
docker run -p 8000:8000 -v ./data:/app/data rag-service
```

### Docker Compose

```bash
docker-compose up -d
```

## Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
make test-cov
```

### Code Quality

```bash
# Lint
make lint

# Format
make format

# Type check
make type-check

# All checks
make check
```

## Project Structure

```
rag-documentation-service/
├── src/rag_service/
│   ├── api/v1/              # API endpoints
│   │   └── endpoints/
│   │       ├── query.py     # Query endpoints (hybrid search)
│   │       └── ingest.py    # Document ingestion
│   ├── core/                # Business logic
│   │   ├── embeddings.py    # Embedding service
│   │   ├── chunker.py       # Document processing
│   │   ├── retriever.py     # Vector store interface
│   │   ├── router.py        # Smart query routing
│   │   └── graph_extractor.py  # Entity extraction
│   ├── infrastructure/      # External adapters
│   │   ├── faiss_store.py   # FAISS implementation
│   │   ├── chroma_store.py  # ChromaDB implementation
│   │   └── neo4j_store.py   # Neo4j graph store
│   ├── ui/                  # Web interface
│   │   └── app.py           # Gradio UI
│   ├── config.py            # Configuration
│   ├── dependencies.py      # Dependency injection
│   └── main.py              # FastAPI app
├── tests/                   # Test suite
├── data/                    # Data directories
├── pyproject.toml          # Python packaging
├── Dockerfile              # Container build
└── docker-compose.yml      # Container orchestration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Run checks (`make check`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [unstructured](https://unstructured.io/) - Document processing


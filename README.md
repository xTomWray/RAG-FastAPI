# RAG Documentation Service

A cross-platform, open-source RAG (Retrieval-Augmented Generation) API service for document retrieval and question answering. Built with FastAPI, FAISS/ChromaDB, and sentence-transformers.

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA 12.1+ for GPU acceleration

### Installation

```bash
# Clone the repository
git clone https://github.com/youruser/rag-documentation-service.git
cd rag-documentation-service

# Create and activate a virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

#### CPU Installation (Default)
```bash
pip install -r requirements.txt
```

#### GPU/CUDA Installation (Recommended for faster embeddings)
```bash
# Windows/Linux with NVIDIA GPU (CUDA 12.1)
pip install -r requirements-cuda.txt

# This installs PyTorch with CUDA support automatically
```

#### Alternative: Manual CUDA Installation
```bash
# First install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

#### Editable Install (for development)
```bash
pip install -e ".[dev]"
```

> **Note:** Using a virtual environment is strongly recommended to avoid dependency conflicts with other Python projects.

### Run the Service

```bash
# Development mode with hot reload
rag-service start --reload

# Production mode with 4 workers
rag-service start --workers 4

# API-only mode (no Gradio UI)
rag-service start --no-ui --workers 4

# Check service status
rag-service status

# Stop the service
rag-service stop

# Alternative: using make (thin wrapper)
make run           # Same as: rag-service start --reload
make run-prod      # Same as: rag-service start --workers 4 --no-ui
```

The service is now available at http://localhost:8080

- **Web UI**: http://localhost:8080 (drag-and-drop interface)
- **API Docs**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

The Web UI is enabled by default and provides:
- 📁 Drag-and-drop file upload
- 📂 Index entire folders
- 🔍 Semantic search with auto/vector/graph/hybrid strategies
- 📦 Collection management
- ℹ️ System status monitoring

### Deployment Patterns

```bash
# All-in-One (Development)
rag-service start --reload

# API-Only (Production Backend)
rag-service start --no-ui --workers 4

# Separate UI (Scaled Frontend)
rag-service start --no-ui --port 8080  # Terminal 1
rag-ui --api-url http://api:8080       # Terminal 2

# Docker
rag-service docker-build
rag-service docker-run
```

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

Settings can be configured via:
1. **Web UI** - Edit settings in the Configuration tab and click "Save" or "Apply & Restart"
2. **config.yaml** - Edit the `config.yaml` file directly (human-readable)
3. **Environment variables** - Override any setting (highest priority)
4. **.env file** - For secrets like passwords

### Using config.yaml (Recommended)

The `config.yaml` file is created automatically and persists your settings between sessions:

```yaml
# config.yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
device: cuda  # auto, cpu, cuda, or mps
chunk_size: 512
vector_store_backend: faiss
enable_graph_rag: true
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
rag-service install --gpu

# Run
rag-service start --reload
```

### Linux

```bash
# Install system dependencies for PDF support
sudo apt install libmagic1 poppler-utils

# Install Python package
pip install -e .

# Run
rag-service start --reload
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
rag-service start --reload
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

# Run container (GUI + API on port 8080)
# --restart unless-stopped enables config changes via UI to trigger restart
docker run -d --restart unless-stopped -p 8080:8080 -v ./data:/app/data --name rag-service rag-service

# Run API-only (no GUI)
docker run -d --restart unless-stopped -p 8080:8080 -e ENABLE_GUI=false -v ./data:/app/data --name rag-service rag-service

# Run with GPU support
docker run -d --restart unless-stopped --gpus all -p 8080:8080 -v ./data:/app/data --name rag-service rag-service
```

### Stop and Manage Containers

```bash
# Stop the running container
docker stop rag-service

# Start a stopped container
docker start rag-service

# Stop and remove the container
docker stop rag-service
docker rm rag-service

# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View container logs
docker logs rag-service

# Follow container logs in real-time
docker logs -f rag-service
```

**Note:** If you're running the service locally (outside Docker) and get a "port already in use" error, stop the Docker container first:
```bash
docker stop rag-service
```

Access:
- **Web UI**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

### Docker Compose

```bash
docker-compose up -d
```

## CLI Reference

The `rag-service` CLI provides all commands for managing the service:

### Service Commands

```bash
rag-service start [OPTIONS]     # Start the RAG service
  --reload, -r                  # Enable hot reload (dev mode)
  --no-ui                       # API-only mode (disable Gradio UI)
  --host, -h TEXT               # Host to bind to
  --port, -p INTEGER            # Port to bind to
  --workers, -w INTEGER         # Number of uvicorn workers

rag-service stop                # Stop the running service
rag-service restart             # Restart the service
rag-service status              # Show service status
```

### Installation Commands

```bash
rag-service install             # Install production dependencies
rag-service install --dev       # Install dev dependencies + pre-commit hooks
rag-service install --gpu       # Install GPU support (CUDA)
rag-service install --all       # Install all optional dependencies
```

### Development Commands

```bash
rag-service test                # Run all tests
rag-service test --unit         # Unit tests only
rag-service test --integration  # Integration tests only
rag-service test --coverage     # With coverage report

rag-service lint                # Run linter (ruff)
rag-service lint --fix          # Auto-fix linting issues

rag-service format              # Format code (black + isort)
rag-service format --check      # Check formatting only

rag-service typecheck           # Run type checker (mypy)
rag-service check               # Run all quality checks
```

### Docker Commands

```bash
rag-service docker-build        # Build Docker image
rag-service docker-build --gpu  # Build with GPU support
rag-service docker-run          # Run in Docker container
rag-service docker-run --gpu    # Run with GPU support
```

### Utility Commands

```bash
rag-service clean               # Clean build artifacts
rag-service clean --data        # Also clean data directories
rag-service --help              # Show all commands
```

## Development

### Setup

```bash
# Install dev dependencies using CLI
rag-service install --dev

# Or using pip directly
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
rag-service test

# Unit tests only
rag-service test --unit

# With coverage
rag-service test --coverage

# Makefile shortcuts (delegates to CLI)
make test        # Same as: rag-service test
make test-unit   # Same as: rag-service test --unit
make test-cov    # Same as: rag-service test --coverage
```

### Code Quality

```bash
# Lint
rag-service lint

# Auto-fix linting issues
rag-service lint --fix

# Format code
rag-service format

# Type check
rag-service typecheck

# All checks (lint + format + typecheck)
rag-service check

# Makefile shortcuts
make lint        # Same as: rag-service lint
make format      # Same as: rag-service format
make check       # Same as: rag-service check
```

## Project Structure

```
rag-documentation-service/
├── src/rag_service/
│   ├── cli/                 # Typer CLI commands
│   │   ├── main.py          # Root CLI app
│   │   ├── commands.py      # Service commands (start/stop/status)
│   │   ├── install_commands.py  # Install command
│   │   ├── dev_commands.py  # Test/lint/format commands
│   │   └── docker_commands.py   # Docker commands
│   ├── api/v1/              # API endpoints
│   │   └── endpoints/
│   │       ├── query.py     # Query endpoints (hybrid search)
│   │       └── ingest.py    # Document ingestion
│   ├── core/                # Business logic
│   │   ├── embeddings.py    # Embedding service
│   │   ├── chunker.py       # Document processing
│   │   ├── retriever.py     # Vector store interface
│   │   ├── router.py        # Smart query routing
│   │   ├── logging.py       # Structured logging (structlog)
│   │   └── async_utils.py   # ThreadPoolExecutor helpers
│   ├── infrastructure/      # External adapters
│   │   ├── faiss_store.py   # FAISS implementation
│   │   ├── chroma_store.py  # ChromaDB implementation
│   │   ├── neo4j_store.py   # Neo4j graph store
│   │   └── cache.py         # Caching layer (memory/redis)
│   ├── middleware/          # Request processing
│   │   └── correlation.py   # Correlation ID middleware
│   ├── ui/                  # Web interface
│   │   └── app.py           # Gradio UI
│   ├── config.py            # Configuration
│   ├── dependencies.py      # Dependency injection
│   └── main.py              # FastAPI app
├── tests/                   # Test suite
├── data/                    # Data directories
├── Makefile                # Thin wrapper for CLI commands
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


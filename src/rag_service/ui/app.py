"""Gradio web interface for the RAG Documentation Service.

Provides a simple drag-and-drop interface for document ingestion and
semantic search queries with hybrid vector/graph retrieval support.
"""

import contextlib
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, get_args

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_UI_PORT = 7860

# Global state for accumulated files (used across Gradio callbacks)
accumulated_files: list[Any] = []


def get_field_constraints(field_name: str) -> dict[str, Any]:
    """Extract field constraints from Settings model (single source of truth).

    This function reads the Pydantic Field metadata from config.py to ensure
    UI components use the same constraints as the configuration system.

    Args:
        field_name: Name of the field in Settings class.

    Returns:
        Dictionary with 'min', 'max', 'default', 'choices', and 'description'.
    """
    from rag_service.config import Settings

    field_info = Settings.model_fields.get(field_name)
    if not field_info:
        return {}

    result = {
        "default": field_info.default,
        "description": field_info.description or "",
    }

    # Extract min/max from Pydantic v2 metadata
    for meta in field_info.metadata:
        if hasattr(meta, "ge"):
            result["min"] = meta.ge
        if hasattr(meta, "gt"):
            result["min"] = meta.gt + 1
        if hasattr(meta, "le"):
            result["max"] = meta.le
        if hasattr(meta, "lt"):
            result["max"] = meta.lt - 1

    # Extract choices from Literal type annotation
    annotation = field_info.annotation
    if (
        annotation is not None
        and hasattr(annotation, "__origin__")
        and annotation.__origin__ is type(None)
    ):
        # Handle Optional types
        pass
    else:
        # Try to get Literal choices
        try:
            args = get_args(annotation)
            if args and all(isinstance(arg, str) for arg in args):
                result["choices"] = list(args)
        except Exception:
            pass

    return result


# Pre-load commonly used constraints (avoids repeated lookups)
FIELD_CONSTRAINTS = {
    "embedding_batch_size": get_field_constraints("embedding_batch_size"),
    "chunk_size": get_field_constraints("chunk_size"),
    "chunk_overlap": get_field_constraints("chunk_overlap"),
    "default_top_k": get_field_constraints("default_top_k"),
    "port": get_field_constraints("port"),
    "device": get_field_constraints("device"),
    "vector_store_backend": get_field_constraints("vector_store_backend"),
    "pdf_strategy": get_field_constraints("pdf_strategy"),
    "graph_store_backend": get_field_constraints("graph_store_backend"),
    "router_mode": get_field_constraints("router_mode"),
    "default_query_strategy": get_field_constraints("default_query_strategy"),
    "entity_extraction_mode": get_field_constraints("entity_extraction_mode"),
    "entity_extraction_domain": get_field_constraints("entity_extraction_domain"),
    "log_level": get_field_constraints("log_level"),
}

# Editable configuration parameters with descriptions and tooltips
CONFIG_SCHEMA = {
    "embedding_model": {
        "choices": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-mistral-7b-instruct",
            "Salesforce/SFR-Embedding-Mistral",
        ],
        "tooltip": "HuggingFace model ID for generating embeddings. Larger models need more VRAM but produce better results. MiniLM: ~90MB (CPU), bge-large: ~1.3GB (8GB+ VRAM), e5-mistral: ~28GB (50GB+ VRAM)",
    },
    "device": {
        "choices": ["auto", "cpu", "cuda", "mps"],
        "tooltip": "Compute device for model inference. 'auto' detects best available (CUDA GPU > Apple MPS > CPU). Use 'cpu' if GPU causes issues.",
    },
    "vector_store_backend": {
        "choices": ["faiss", "chroma"],
        "tooltip": "Vector database for storing embeddings. FAISS: Fastest, best for large datasets. ChromaDB: Simpler setup, built-in filtering.",
    },
    "chunk_size": {
        "tooltip": "Maximum characters per document chunk. Smaller chunks = more precise retrieval but less context. Larger chunks = more context but may include irrelevant text. Typical: 256-1024.",
    },
    "chunk_overlap": {
        "tooltip": "Characters of overlap between consecutive chunks. Helps preserve context across chunk boundaries. Typical: 10-20% of chunk size.",
    },
    "enable_graph_rag": {
        "tooltip": "Enable knowledge graph features for relational queries. Extracts entities and relationships from documents. Best for protocols, state machines, and structured data.",
    },
    "graph_store_backend": {
        "choices": ["memory", "neo4j"],
        "tooltip": "Graph database backend. 'memory': Fast, no setup, lost on restart. 'neo4j': Persistent, scalable, requires Neo4j server.",
    },
    "router_mode": {
        "choices": ["pattern", "llm"],
        "tooltip": "How queries are classified for routing. 'pattern': Fast rule-based matching. 'llm': Uses Ollama for intelligent classification (requires Ollama running).",
    },
    "entity_extraction_mode": {
        "choices": ["rule_based", "llm"],
        "tooltip": "How entities/relationships are extracted from text. 'rule_based': Fast regex patterns. 'llm': Uses Ollama for better extraction (slower, requires Ollama).",
    },
    "log_level": {
        "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
        "tooltip": "Logging verbosity. DEBUG: All details. INFO: Normal operation. WARNING: Potential issues. ERROR: Only errors.",
    },
}


def escape_html(text: str) -> str:
    """Escape HTML special characters in text.

    Args:
        text: The text to escape.

    Returns:
        HTML-escaped text.
    """
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def find_available_port(start_port: int = 7860, max_attempts: int = 100) -> int:
    """Find an available port starting from the given port.

    Args:
        start_port: Port number to start searching from.
        max_attempts: Maximum number of ports to try.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no available port is found within max_attempts.
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
                logger.info(f"Found available port: {port}")
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def check_api_health(api_url: str) -> tuple[bool, str]:
    """Check if the RAG API service is running.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Tuple of (is_healthy, status_message).
    """
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        if response.status_code == 200:
            return True, "✅ API is running"
        return False, f"⚠️ API returned status {response.status_code}"
    except httpx.ConnectError:
        return False, "❌ Cannot connect to API. Start it with: python -m rag_service"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def get_system_info(api_url: str) -> dict[str, Any]:
    """Fetch system information from the API.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        System info dictionary or error dict.
    """
    try:
        response = httpx.get(f"{api_url}/info", timeout=5.0)
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def load_current_config() -> dict[str, Any]:
    """Load current configuration from config.yaml and settings."""
    from rag_service.config import get_config_file_path, get_settings

    try:
        settings = get_settings()
        config_path = get_config_file_path()
        return {
            # Embedding & Processing
            "embedding_model": settings.embedding_model,
            "device": settings.device,
            "embedding_batch_size": settings.embedding_batch_size,
            # Document Processing
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "pdf_strategy": settings.pdf_strategy,
            # Vector Store
            "vector_store_backend": settings.vector_store_backend,
            "faiss_index_dir": str(settings.faiss_index_dir),
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "default_collection": settings.default_collection,
            # API
            "host": settings.host,
            "port": settings.port,
            "api_prefix": settings.api_prefix,
            "cors_origins": ",".join(settings.cors_origins),
            "log_level": settings.log_level,
            # GraphRAG
            "enable_graph_rag": settings.enable_graph_rag,
            "graph_store_backend": settings.graph_store_backend,
            "neo4j_uri": settings.neo4j_uri,
            "neo4j_user": settings.neo4j_user,
            "neo4j_database": settings.neo4j_database,
            # Query & Extraction
            "router_mode": settings.router_mode,
            "default_query_strategy": settings.default_query_strategy,
            "default_top_k": settings.default_top_k,
            "entity_extraction_mode": settings.entity_extraction_mode,
            "entity_extraction_domain": settings.entity_extraction_domain,
            "ollama_model": settings.ollama_model,
            # Meta
            "_config_file": str(config_path) if config_path else "config.yaml (will be created)",
        }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def save_config_to_yaml(config: dict[str, Any]) -> str:
    """Save configuration to config.yaml file.

    Args:
        config: Configuration dictionary.

    Returns:
        Status message.
    """
    from rag_service.config import get_settings, save_yaml_config

    try:
        # Get current full settings and update with new values
        current_settings = get_settings()
        full_config = current_settings.to_yaml_dict()

        # All UI-editable fields
        ui_fields = [
            "embedding_model",
            "device",
            "embedding_batch_size",
            "chunk_size",
            "chunk_overlap",
            "pdf_strategy",
            "vector_store_backend",
            "faiss_index_dir",
            "chroma_persist_dir",
            "default_collection",
            "host",
            "port",
            "api_prefix",
            "log_level",
            "enable_graph_rag",
            "graph_store_backend",
            "neo4j_uri",
            "neo4j_user",
            "neo4j_database",
            "router_mode",
            "default_query_strategy",
            "default_top_k",
            "entity_extraction_mode",
            "entity_extraction_domain",
            "ollama_model",
        ]

        for field in ui_fields:
            if field in config:
                value = config[field]
                # Handle cors_origins specially (convert comma-separated string to list)
                if field == "cors_origins" and isinstance(value, str):
                    value = [x.strip() for x in value.split(",") if x.strip()]
                full_config[field] = value

        # Handle cors_origins if present
        if "cors_origins" in config:
            value = config["cors_origins"]
            if isinstance(value, str):
                full_config["cors_origins"] = [x.strip() for x in value.split(",") if x.strip()]
            else:
                full_config["cors_origins"] = value

        # Save to YAML
        config_path = save_yaml_config(full_config)
        return f"✅ Configuration saved to {config_path}\n⚠️ Restart service to apply changes"

    except PermissionError as e:
        return f"❌ Permission denied: {e}"
    except Exception as e:
        logger.exception("Failed to save config")
        return f"❌ Error saving config: {e}"


def apply_and_restart(config: dict[str, Any]) -> str:
    """Apply configuration and restart the service.

    Saves config to config.yaml, then triggers an in-place restart using
    exit code 42 (RESTART_EXIT_CODE). The service wrapper in __main__.py
    catches this and restarts in the same console window.

    Args:
        config: Configuration dictionary.

    Returns:
        Status message (shown briefly before restart).
    """
    import threading
    import time

    # Import the restart exit code
    from rag_service.__main__ import RESTART_EXIT_CODE

    try:
        logger.info("Applying configuration and preparing restart...")

        # Step 1: Save to config.yaml for persistence
        save_result = save_config_to_yaml(config)
        if save_result.startswith("❌"):
            return save_result

        # Step 2: Get the current service's port from config
        from rag_service.config import get_settings

        settings = get_settings()
        port = config.get("port", settings.port)

        # Step 3: Schedule restart after response is sent
        def delayed_restart() -> None:
            time.sleep(1.0)  # Brief wait for response to be sent
            logger.info("Initiating in-place service restart (exit code 42)...")
            # Exit with restart code - the wrapper loop will restart us
            os._exit(RESTART_EXIT_CODE)

        threading.Thread(target=delayed_restart, daemon=False).start()

        return (
            "✅ Configuration saved to config.yaml!\n"
            "🔄 Service restarting in same console...\n"
            "⏳ Please wait ~10-30 seconds for reload.\n"
            "🔃 Refresh this page when ready.\n\n"
            f"💡 Service will restart on port {port}"
        )

    except Exception as e:
        logger.exception("Failed to apply and restart")
        return f"❌ Error: {e}"


def get_config_display() -> tuple[Any, ...]:
    """Get current config values for UI display.

    Returns:
        Tuple of config values in order matching UI components.
    """
    config = load_current_config()
    return (
        # Row 1: Embedding
        config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        config.get("device", "auto"),
        config.get("embedding_batch_size", 32),
        # Row 2: Document Processing
        config.get("chunk_size", 512),
        config.get("chunk_overlap", 50),
        config.get("pdf_strategy", "fast"),
        # Row 3: Vector Store
        config.get("vector_store_backend", "faiss"),
        config.get("faiss_index_dir", "./data/index"),
        config.get("chroma_persist_dir", "./data/chroma"),
        config.get("default_collection", "documents"),
        # Row 4: API
        config.get("host", "0.0.0.0"),
        config.get("port", 8080),
        config.get("api_prefix", "/api/v1"),
        config.get("cors_origins", "http://localhost:3000,http://localhost:8080"),
        config.get("log_level", "INFO"),
        # Row 5: GraphRAG
        config.get("enable_graph_rag", True),
        config.get("graph_store_backend", "memory"),
        config.get("neo4j_uri", "bolt://localhost:7687"),
        config.get("neo4j_user", "neo4j"),
        config.get("neo4j_database", "neo4j"),
        # Row 6: Query & Extraction
        config.get("router_mode", "pattern"),
        config.get("default_query_strategy", "vector"),
        config.get("default_top_k", 5),
        config.get("entity_extraction_mode", "rule_based"),
        config.get("entity_extraction_domain", "general"),
        config.get("ollama_model", "llama3.2"),
    )


def format_file_list_html(files: list[Any]) -> str:
    """Format a list of files for HTML display with scrolling.

    Args:
        files: List of file objects from Gradio.

    Returns:
        HTML string with file list in a scrollable container.
    """
    if not files:
        return '<div class="file-list-container"><em>No files selected</em></div>'

    file_items = []
    for i, file in enumerate(files, 1):
        file_path = file.name if hasattr(file, "name") else str(file)
        file_name = escape_html(Path(file_path).name)
        file_items.append(f"<p>{i}. {file_name}</p>")

    count = len(files)
    header = f"<p><strong>{count} file{'s' if count != 1 else ''} queued:</strong></p>"
    content = header + "".join(file_items)
    return f'<div class="file-list-container">{content}</div>'


def merge_files_and_return_list(new_files: list[Any]) -> list[Any]:
    """Add new files to the accumulated list, avoiding duplicates.

    Args:
        new_files: List of newly added file objects.

    Returns:
        The full accumulated files list (to update the File component).
    """
    global accumulated_files

    if not new_files:
        return accumulated_files

    # Convert existing files to paths for comparison
    existing_paths = set()
    for f in accumulated_files:
        file_path = f.name if hasattr(f, "name") else str(f)
        existing_paths.add(file_path)

    # Add new files that aren't already in the list
    for new_file in new_files:
        file_path = new_file.name if hasattr(new_file, "name") else str(new_file)
        if file_path not in existing_paths:
            accumulated_files.append(new_file)
            existing_paths.add(file_path)

    return accumulated_files


def clear_file_queue() -> None:
    """Clear all accumulated files.

    Returns:
        None to clear the File component.
    """
    global accumulated_files
    accumulated_files.clear()
    return None


def get_queue_count() -> str:
    """Get the current queue count as a status string."""
    global accumulated_files
    count = len(accumulated_files)
    if count == 0:
        return "📂 **Drop files here to build your queue**"
    return f"📂 **{count} file{'s' if count != 1 else ''} in queue** - drop more to add"


def safe_list_collections(api_url: str) -> tuple[str, str]:
    """Safe wrapper for list_collections with error handling.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Tuple of (collections display text, status message).
    """
    try:
        result = list_collections(api_url)
        return result, "✅ Collections loaded successfully"
    except Exception as e:
        error_msg = f"⚠️ Could not load collections: {str(e)}"
        return error_msg, f"❌ Error: {str(e)}"


def safe_get_system_info(api_url: str) -> tuple[dict[str, Any], str]:
    """Safe wrapper for get_system_info with error handling.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Tuple of (system info dict, status message).
    """
    try:
        info = get_system_info(api_url)
        return info, "✅ System info loaded successfully"
    except Exception as e:
        error_dict = {"error": f"Could not load system info: {str(e)}"}
        return error_dict, f"❌ Error: {str(e)}"


def ingest_files(
    files: list[Any],
    collection: str,
    api_url: str,
) -> tuple[str, ...]:
    """Ingest uploaded files into the RAG service.

    This function copies files to a stable location before processing to avoid
    race conditions with Gradio's temporary file cleanup. Files are processed
    concurrently for better performance.

    Args:
        files: List of uploaded file objects from Gradio.
        collection: Target collection name.
        api_url: Base URL of the RAG API service.

    Returns:
        Tuple of (status message).
    """
    import shutil
    import tempfile
    import uuid
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not files:
        return ("⚠️ No files selected", "")

    if not collection.strip():
        collection = "documents"

    # Create a stable temp directory for this upload session
    # This prevents race conditions with Gradio's temp file cleanup
    session_id = uuid.uuid4().hex[:8]
    stable_dir = Path(tempfile.gettempdir()) / f"rag_upload_{session_id}"
    stable_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy all files to stable location FIRST (before any processing)
    file_mapping = {}  # Maps stable_path -> original_name
    copy_errors = []

    for file in files:
        try:
            src_path = Path(file.name if hasattr(file, "name") else str(file))
            original_name = src_path.name

            # Create unique stable path (handle duplicate names)
            stable_path = stable_dir / f"{uuid.uuid4().hex[:8]}_{original_name}"

            # Copy file to stable location
            shutil.copy2(src_path, stable_path)
            file_mapping[stable_path] = original_name

        except Exception as e:
            original_name = Path(str(file)).name if file else "unknown"
            copy_errors.append(f"❌ {original_name}: Failed to copy - {str(e)}")
            logger.warning(f"Failed to copy file {file}: {e}")

    if not file_mapping:
        # All copies failed - cleanup and return
        shutil.rmtree(stable_dir, ignore_errors=True)
        error_msg = "\n".join(copy_errors) if copy_errors else "❌ No files could be copied"
        return (error_msg,)

    # Step 2: Process files concurrently from stable location
    results = list(copy_errors)  # Start with any copy errors
    success_count = 0
    total_chunks = 0

    def process_single_file(stable_path: Path, original_name: str) -> tuple[str, int, bool]:
        """Process a single file and return (result_msg, chunks, success)."""
        try:
            response = httpx.post(
                f"{api_url}/api/v1/ingest/file",
                json={"path": str(stable_path), "collection": collection},
                timeout=120.0,
            )

            if response.status_code == 200:
                data = response.json()
                chunks = data.get("documents_processed", 0)
                return (f"✅ {original_name}: {chunks} chunks", chunks, True)
            else:
                error = response.json().get("detail", "Unknown error")
                return (f"❌ {original_name}: {error}", 0, False)

        except httpx.TimeoutException:
            return (f"❌ {original_name}: Request timeout (>120s)", 0, False)
        except httpx.ConnectError:
            return (f"❌ {original_name}: Cannot connect to API", 0, False)
        except Exception as e:
            return (f"❌ {original_name}: {str(e)}", 0, False)

    # Use thread pool for concurrent processing (max 4 concurrent to avoid overload)
    max_workers = min(4, len(file_mapping))

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_file, stable_path, original_name): original_name
                for stable_path, original_name in file_mapping.items()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result_msg, chunks, success = future.result()
                    results.append(result_msg)
                    if success:
                        success_count += 1
                        total_chunks += chunks
                except Exception as e:
                    original_name = futures[future]
                    results.append(f"❌ {original_name}: Processing error - {str(e)}")

    finally:
        # Step 3: Clean up stable temp directory
        with contextlib.suppress(Exception):
            shutil.rmtree(stable_dir, ignore_errors=True)

    # Build summary
    total_files = len(files)
    summary = f"\n{'─' * 40}\n📊 Summary: {success_count}/{total_files} files, {total_chunks:,} total chunks"
    status_message = "\n".join(results) + summary

    # Clear the global accumulated files list after upload so user can start fresh
    global accumulated_files
    accumulated_files.clear()

    return (status_message, "")


def ingest_directory(
    directory_path: str,
    collection: str,
    recursive: bool,
    api_url: str,
) -> str:
    """Ingest all documents from a directory.

    Args:
        directory_path: Path to the directory.
        collection: Target collection name.
        recursive: Whether to process subdirectories.
        api_url: Base URL of the RAG API service.

    Returns:
        Status message with results.
    """
    if not directory_path.strip():
        return "⚠️ Please enter a directory path"

    if not collection.strip():
        collection = "documents"

    try:
        response = httpx.post(
            f"{api_url}/api/v1/ingest/directory",
            json={
                "path": directory_path,
                "collection": collection,
                "recursive": recursive,
            },
            timeout=300.0,
        )

        if response.status_code == 200:
            data = response.json()
            return (
                f"✅ Ingestion complete!\n"
                f"📁 Files processed: {data.get('files_processed', 0)}\n"
                f"📄 Document chunks: {data.get('documents_processed', 0)}\n"
                f"📦 Collection: {collection}"
            )
        else:
            error = response.json().get("detail", "Unknown error")
            return f"❌ Error: {error}"

    except httpx.ConnectError:
        return "❌ Cannot connect to API. Is the service running?"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def query_documents(
    question: str,
    collection: str,
    top_k: int,
    strategy: str,
    api_url: str,
) -> tuple[str, str, str]:
    """Query the RAG service for relevant documents.

    Args:
        question: The search query.
        collection: Collection to search in.
        top_k: Number of results to return.
        strategy: Search strategy (auto, vector, graph, hybrid).
        api_url: Base URL of the RAG API service.

    Returns:
        Tuple of (formatted_results, sources_text, metadata_json).
    """
    if not question.strip():
        return "⚠️ Please enter a question", "", "{}"

    if not collection.strip():
        collection = "documents"

    try:
        # Use hybrid endpoint for full features
        response = httpx.post(
            f"{api_url}/api/v1/query/hybrid",
            json={
                "question": question,
                "collection": collection,
                "top_k": top_k,
                "strategy": strategy,
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            chunks = data.get("chunks", [])
            graph_context = data.get("graph_context", [])
            strategy_used = data.get("strategy_used", "unknown")
            reasoning = data.get("router_reasoning", "")

            # Format results
            results_parts = [f"🔍 Strategy: {strategy_used}"]
            if reasoning:
                results_parts.append(f"💡 {reasoning}\n")

            if chunks:
                results_parts.append(f"📄 Found {len(chunks)} relevant chunks:\n")
                for i, chunk in enumerate(chunks, 1):
                    score = chunk.get("score", 0)
                    text = chunk.get("text", "")[:500]
                    source = chunk.get("metadata", {}).get("source", "Unknown")
                    results_parts.append(
                        f"{'─' * 40}\n"
                        f"**[{i}] Score: {score:.3f}** | Source: {Path(source).name}\n\n"
                        f"{text}{'...' if len(chunk.get('text', '')) > 500 else ''}\n"
                    )

            if graph_context:
                results_parts.append(f"\n🔗 Graph Context ({len(graph_context)} items):\n")
                for item in graph_context[:10]:
                    entity = item.get("entity", "")
                    rel = item.get("relationship", "")
                    target = item.get("connected_to", "")
                    if rel and target:
                        results_parts.append(f"  • {entity} —[{rel}]→ {target}")
                    else:
                        results_parts.append(f"  • {entity} ({item.get('entity_type', '')})")

            results_text = "\n".join(results_parts)

            # Format sources
            sources = data.get("sources", [])
            sources_text = "\n".join(f"📁 {s}" for s in sources) if sources else "No sources found"

            # Metadata
            metadata = {
                "token_estimate": data.get("token_estimate", 0),
                "strategy_used": strategy_used,
                "chunks_found": len(chunks),
                "graph_items": len(graph_context),
            }

            return results_text, sources_text, json.dumps(metadata, indent=2)

        elif response.status_code == 404:
            return f"⚠️ Collection '{collection}' not found. Ingest documents first.", "", "{}"
        else:
            error = response.json().get("detail", "Unknown error")
            return f"❌ Error: {error}", "", "{}"

    except httpx.ConnectError:
        return "❌ Cannot connect to API. Is the service running?", "", "{}"
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "{}"


def get_collection_choices(api_url: str) -> list[str]:
    """Get list of collection names for dropdown.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        List of collection names, with "documents" as default if empty.
    """
    try:
        response = httpx.get(f"{api_url}/api/v1/collections", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            vector_cols = data.get("vector_collections", [])

            # Extract collection names
            names = [col.get("name", "") for col in vector_cols if col.get("name")]

            # Always include "documents" as an option
            if "documents" not in names:
                names.insert(0, "documents")

            return sorted(set(names)) if names else ["documents"]
    except Exception:
        pass

    return ["documents"]


def refresh_collection_dropdown(api_url: str) -> dict[str, Any]:
    """Refresh collection dropdown choices.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Gradio update dict with new choices.
    """
    choices = get_collection_choices(api_url)
    # Return update with new choices, keeping current value if valid
    return gr.update(choices=choices, value=choices[0] if choices else "documents")  # type: ignore[no-any-return]


def list_collections(api_url: str) -> str:
    """List all available collections.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Formatted list of collections as markdown.
    """
    try:
        response = httpx.get(f"{api_url}/api/v1/collections", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            vector_cols = data.get("vector_collections", [])
            graph_cols = data.get("graph_collections", [])

            vector_cols = sorted(
                vector_cols,
                key=lambda col: col.get("name", "").lower(),
            )
            graph_cols = sorted(
                graph_cols,
                key=lambda col: col.get("name", "").lower(),
            )

            # Build markdown with proper formatting (double newlines for line breaks)
            parts = ["## 📦 Vector Collections\n"]

            if vector_cols:
                for col in vector_cols:
                    name = col.get("name", "unknown")
                    count = col.get("document_count", 0)
                    # Format large numbers with commas for readability
                    count_formatted = f"{count:,}"
                    parts.append(f"| **{name}** | {count_formatted} chunks |")

                # Add a table header at the beginning
                parts.insert(1, "| Collection | Indexed Chunks |")
                parts.insert(2, "|:-----------|---------------:|")
            else:
                parts.append("*No vector collections found*\n")

            if graph_cols:
                parts.append("\n---\n")
                parts.append("## 🔗 Graph Collections\n")
                parts.append("| Collection | Nodes | Relationships |")
                parts.append("|:-----------|------:|--------------:|")
                for col in graph_cols:
                    name = col.get("name", "unknown")
                    nodes = col.get("total_nodes", 0)
                    rels = col.get("total_relationships", 0)
                    parts.append(f"| **{name}** | {nodes:,} | {rels:,} |")

            return "\n".join(parts)
        return "❌ Failed to fetch collections"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def delete_collection(collection: str, api_url: str) -> str:
    """Delete a collection.

    Args:
        collection: Name of the collection to delete.
        api_url: Base URL of the RAG API service.

    Returns:
        Status message.
    """
    if not collection.strip():
        return "⚠️ Please enter a collection name"

    try:
        response = httpx.delete(
            f"{api_url}/api/v1/collections/{collection}",
            timeout=30.0,
        )
        if response.status_code == 200:
            return f"✅ Collection '{collection}' deleted"
        else:
            error = response.json().get("detail", "Unknown error")
            return f"❌ Error: {error}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_ui(api_url: str = DEFAULT_API_URL) -> gr.Blocks:
    """Create the Gradio interface.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Configured Gradio Blocks interface.
    """
    # Load config for default values
    from rag_service.config import get_settings

    settings = get_settings()
    default_top_k = settings.default_top_k

    # Note: Theme is set via app.theme after creation to avoid Gradio 6.0 deprecation warning
    # when using mount_gradio_app (which doesn't support launch() parameters)
    with gr.Blocks(
        title="RAG Documentation Service",
    ) as app:
        # Set theme after Blocks creation (Gradio 6.0 compatible)
        app.theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )

        # Add custom CSS for file list scrolling (injected via HTML)
        gr.HTML(
            value="""
            <style>
                .file-list-container {
                    max-height: 220px;
                    overflow-y: auto;
                    padding: 8px 12px;
                    border: 1px solid var(--border-color-primary, #ddd);
                    border-radius: 6px;
                    background: var(--background-fill-secondary, #f9f9f9);
                    font-family: ui-monospace, monospace;
                    font-size: 0.85em;
                    line-height: 1.4;
                }
                .file-list-container p {
                    margin: 2px 0;
                }
            </style>
            """,
            visible=False,
        )

        gr.Markdown(
            """
            # 📚 RAG Documentation Service

            Upload documents, build a knowledge base, and search with hybrid vector + graph retrieval.
            """
        )

        # Hidden API URL input (used by other components)
        api_url_input = gr.Textbox(value=api_url, visible=False)

        # Note: accumulated_files is defined as a module-level global variable
        # to maintain state across Gradio callbacks

        with gr.Tabs():
            # Tab 1: Document Upload
            with gr.Tab("📁 Upload Documents") as upload_tab:
                with gr.Row():
                    # Left column: Drop zone + queue list
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 File Queue")
                        # Compact drop zone - just for receiving files
                        file_dropzone = gr.File(
                            file_count="multiple",
                            file_types=[
                                ".pdf",
                                ".md",
                                ".txt",
                                ".xml",
                                ".html",
                                ".docx",
                                ".py",
                                ".js",
                                ".json",
                                ".yaml",
                            ],
                            label="Drop files here (auto-adds to queue below)",
                            height=80,
                        )
                        # Queue display - shows accumulated files
                        queue_display = gr.HTML(
                            value='<div class="file-list-container"><em>Queue empty - drop files above</em></div>',
                        )
                        with gr.Row():
                            clear_queue_btn = gr.Button(
                                "🗑️ Clear Queue", variant="secondary", size="sm"
                            )
                        with gr.Row():
                            upload_collection = gr.Dropdown(
                                choices=["documents"],
                                value="documents",
                                label="Collection",
                                scale=3,
                                allow_custom_value=True,
                                info="Select or type a collection name",
                            )
                            refresh_upload_collections_btn = gr.Button(
                                "🔄",
                                size="sm",
                                scale=0,
                                min_width=40,
                            )
                        upload_btn = gr.Button("📤 Upload & Index Files", variant="primary")

                    # Right column: Folder indexing
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Index a Folder")
                        dir_path = gr.Textbox(
                            label="Directory Path",
                            placeholder="C:\\path\\to\\documents or /path/to/documents",
                            info="Full path to document folder",
                        )
                        dir_recursive = gr.Checkbox(
                            value=True,
                            label="Include subdirectories",
                        )
                        with gr.Row():
                            dir_collection = gr.Dropdown(
                                choices=["documents"],
                                value="documents",
                                label="Collection",
                                scale=3,
                                allow_custom_value=True,
                                info="Select or type a collection name",
                            )
                            refresh_dir_collections_btn = gr.Button(
                                "🔄",
                                size="sm",
                                scale=0,
                                min_width=40,
                            )
                        dir_btn = gr.Button("📂 Index Folder", variant="primary")

                upload_output = gr.Textbox(
                    label="Results",
                    lines=8,
                    interactive=False,
                )

                # When files are dropped, add to queue and update display (don't touch dropzone)
                def on_files_dropped(files: Any) -> tuple[str, None]:
                    """Handle newly dropped files and reset the dropzone value so users can drop again."""
                    if not files:
                        return format_file_list_html(accumulated_files), None
                    merge_files_and_return_list(files)
                    # Return None for the File component to clear it (per Gradio docs)
                    return format_file_list_html(accumulated_files), None

                # Upload accumulated files to the API
                def upload_queued_files(
                    collection: str, api_url: str
                ) -> tuple[str, str, str, None]:
                    if not accumulated_files:
                        return (
                            "⚠️ No files in queue",
                            "",
                            format_file_list_html(accumulated_files),
                            None,
                        )
                    result = ingest_files(accumulated_files, collection, api_url)
                    # Append queue HTML and clear the dropzone value
                    return result + (format_file_list_html(accumulated_files), None)  # type: ignore[return-value]

                # Clear the queue
                def clear_queue() -> tuple[str, None]:
                    clear_file_queue()
                    # Clear both the queue display and the File component
                    return format_file_list_html(accumulated_files), None

                # Use both upload AND change events to catch file additions
                file_dropzone.upload(
                    fn=on_files_dropped,
                    inputs=[file_dropzone],
                    outputs=[queue_display, file_dropzone],
                )
                file_dropzone.change(
                    fn=on_files_dropped,
                    inputs=[file_dropzone],
                    outputs=[queue_display, file_dropzone],
                )

                # Clear queue button
                clear_queue_btn.click(
                    fn=clear_queue,
                    inputs=[],
                    outputs=[queue_display, file_dropzone],
                )

                # Upload button
                upload_btn.click(
                    fn=upload_queued_files,
                    inputs=[upload_collection, api_url_input],
                    outputs=[upload_output, queue_display, file_dropzone],
                )

                dir_btn.click(
                    fn=ingest_directory,
                    inputs=[dir_path, dir_collection, dir_recursive, api_url_input],
                    outputs=[upload_output],
                )

                # Refresh collection dropdowns in Upload tab
                refresh_upload_collections_btn.click(
                    fn=refresh_collection_dropdown,
                    inputs=[api_url_input],
                    outputs=[upload_collection],
                )
                refresh_dir_collections_btn.click(
                    fn=refresh_collection_dropdown,
                    inputs=[api_url_input],
                    outputs=[dir_collection],
                )

                # Auto-load collections when Upload tab is selected (updates both dropdowns)
                def refresh_both_upload_dropdowns(api_url: str) -> tuple[Any, Any]:
                    """Refresh both collection dropdowns in Upload tab."""
                    result = refresh_collection_dropdown(api_url)
                    return result, result

                upload_tab.select(
                    fn=refresh_both_upload_dropdowns,
                    inputs=[api_url_input],
                    outputs=[upload_collection, dir_collection],
                )

            # Tab 2: Search
            with gr.Tab("🔍 Search") as search_tab:
                with gr.Row(), gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="What would you like to know?",
                        lines=2,
                    )
                    with gr.Row():
                        search_collection = gr.Dropdown(
                            choices=["documents"],
                            value="documents",
                            label="Collection",
                            scale=2,
                            allow_custom_value=True,  # Allow typing custom names
                            info="Select or type a collection name",
                        )
                        refresh_collections_btn = gr.Button(
                            "🔄",
                            size="sm",
                            scale=0,
                            min_width=40,
                        )
                        top_k = gr.Slider(
                            minimum=FIELD_CONSTRAINTS["default_top_k"].get("min", 1),
                            maximum=FIELD_CONSTRAINTS["default_top_k"].get("max", 100),
                            value=default_top_k,
                            step=1,
                            label="Results (top_k)",
                            scale=1,
                        )
                        # Strategy includes "auto" which routes dynamically
                        strategy_choices = ["auto"] + FIELD_CONSTRAINTS[
                            "default_query_strategy"
                        ].get("choices", ["vector", "graph", "hybrid"])
                        strategy = gr.Dropdown(
                            choices=strategy_choices,
                            value="auto",
                            label="Strategy",
                            scale=1,
                        )
                    search_btn = gr.Button("🔍 Search", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        results_output = gr.Markdown(label="Results")
                    with gr.Column(scale=1):
                        sources_output = gr.Textbox(
                            label="Sources",
                            lines=5,
                            interactive=False,
                        )
                        metadata_output = gr.Code(
                            label="Metadata",
                            language="json",
                        )

                search_btn.click(
                    fn=query_documents,
                    inputs=[question_input, search_collection, top_k, strategy, api_url_input],
                    outputs=[results_output, sources_output, metadata_output],
                )
                question_input.submit(
                    fn=query_documents,
                    inputs=[question_input, search_collection, top_k, strategy, api_url_input],
                    outputs=[results_output, sources_output, metadata_output],
                )

                # Refresh collections dropdown
                refresh_collections_btn.click(
                    fn=refresh_collection_dropdown,
                    inputs=[api_url_input],
                    outputs=[search_collection],
                )

                # Auto-load collections when Search tab is selected
                search_tab.select(
                    fn=refresh_collection_dropdown,
                    inputs=[api_url_input],
                    outputs=[search_collection],
                )

            # Tab 3: Collections
            with gr.Tab("📦 Collections"):
                collections_display = gr.Markdown(
                    value="## 📦 Vector Collections\n\n*Click 'Load Collections' to view indexed data*"
                )

                with gr.Row():
                    load_collections_btn = gr.Button(
                        "📦 Load Collections", variant="secondary", size="sm"
                    )
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                collections_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Row():
                    delete_name = gr.Textbox(
                        label="Collection to Delete",
                        placeholder="Enter collection name",
                        scale=3,
                    )
                    delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                delete_output = gr.Textbox(label="Status", interactive=False)

                load_collections_btn.click(
                    fn=safe_list_collections,
                    inputs=[api_url_input],
                    outputs=[collections_display, collections_status],
                )
                refresh_btn.click(
                    fn=safe_list_collections,
                    inputs=[api_url_input],
                    outputs=[collections_display, collections_status],
                )
                delete_btn.click(
                    fn=delete_collection,
                    inputs=[delete_name, api_url_input],
                    outputs=[delete_output],
                )

            # Tab 4: Configuration (Compact layout with all parameters)
            with gr.Tab("⚙️ Configuration"):
                # Get initial config values
                initial_config = load_current_config()

                # Common embedding models for dropdown
                EMBEDDING_MODEL_CHOICES = [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5",
                    "BAAI/bge-large-en-v1.5",
                    "intfloat/e5-small-v2",
                    "intfloat/e5-base-v2",
                    "intfloat/e5-large-v2",
                    "thenlper/gte-small",
                    "thenlper/gte-base",
                    "thenlper/gte-large",
                ]

                # === EMBEDDING SECTION ===
                with gr.Accordion("🧠 Embedding Model", open=True):
                    gr.Markdown(
                        "*Configure the embedding model for semantic search. Larger models = better quality but more VRAM.*",
                        elem_classes=["help-text"],
                    )
                    with gr.Row():
                        cfg_embedding_model = gr.Dropdown(
                            choices=EMBEDDING_MODEL_CHOICES,
                            value=initial_config.get(
                                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                            ),
                            label="Model",
                            allow_custom_value=True,
                            scale=3,
                            info="Select or type any HuggingFace model ID",
                        )
                        cfg_device = gr.Dropdown(
                            choices=["auto", "cpu", "cuda", "mps"],
                            value=initial_config.get("device", "auto"),
                            label="Device",
                            scale=1,
                            info="auto=detect GPU, cuda=NVIDIA, mps=Apple Silicon",
                        )
                        cfg_embedding_batch_size = gr.Number(
                            value=initial_config.get("embedding_batch_size", 32),
                            label="Batch Size",
                            minimum=1,
                            maximum=2048,
                            scale=1,
                            info="Higher = faster but more VRAM",
                        )

                # === DOCUMENT PROCESSING SECTION ===
                with gr.Accordion("📄 Document Processing", open=True):
                    gr.Markdown(
                        "*Control how documents are split into chunks for indexing.*",
                        elem_classes=["help-text"],
                    )
                    with gr.Row():
                        cfg_chunk_size = gr.Number(
                            value=initial_config.get("chunk_size", 512),
                            label="Chunk Size",
                            minimum=100,
                            maximum=4096,
                            info="Characters per chunk (smaller=precise, larger=more context)",
                        )
                        cfg_chunk_overlap = gr.Number(
                            value=initial_config.get("chunk_overlap", 50),
                            label="Overlap",
                            minimum=0,
                            maximum=500,
                            info="Characters overlap between chunks",
                        )
                        cfg_pdf_strategy = gr.Dropdown(
                            choices=["fast", "hi_res", "ocr_only"],
                            value=initial_config.get("pdf_strategy", "fast"),
                            label="PDF Strategy",
                            info="fast=quick, hi_res=complex layouts, ocr_only=scanned docs",
                        )

                # === VECTOR STORE SECTION ===
                with gr.Accordion("💾 Vector Store", open=True):
                    gr.Markdown(
                        "*Choose where embeddings are stored. FAISS is faster, ChromaDB has built-in filtering.*",
                        elem_classes=["help-text"],
                    )
                    with gr.Row():
                        cfg_vector_store = gr.Dropdown(
                            choices=["faiss", "chroma"],
                            value=initial_config.get("vector_store_backend", "faiss"),
                            label="Backend",
                            info="faiss=fastest, chroma=simpler filtering",
                        )
                        cfg_faiss_index_dir = gr.Textbox(
                            value=initial_config.get("faiss_index_dir", "./data/index"),
                            label="FAISS Index Dir",
                            info="Path to store FAISS indices",
                        )
                        cfg_chroma_persist_dir = gr.Textbox(
                            value=initial_config.get("chroma_persist_dir", "./data/chroma"),
                            label="Chroma Dir",
                            info="Path to store ChromaDB data",
                        )
                        cfg_default_collection = gr.Textbox(
                            value=initial_config.get("default_collection", "documents"),
                            label="Default Collection",
                            info="Name for default document collection",
                        )

                # === API SECTION ===
                with gr.Accordion("🌐 API Server", open=False):
                    gr.Markdown(
                        "*Network settings for the API server. Changes require restart.*",
                        elem_classes=["help-text"],
                    )
                    with gr.Row():
                        cfg_host = gr.Textbox(
                            value=initial_config.get("host", "0.0.0.0"),
                            label="Host",
                            scale=1,
                            info="0.0.0.0=all interfaces, 127.0.0.1=localhost only",
                        )
                        cfg_port = gr.Number(
                            value=initial_config.get("port", 8080),
                            label="Port",
                            minimum=1,
                            maximum=65535,
                            scale=1,
                            info="API server port",
                        )
                        cfg_api_prefix = gr.Textbox(
                            value=initial_config.get("api_prefix", "/api/v1"),
                            label="API Prefix",
                            scale=1,
                            info="URL prefix for API routes",
                        )
                        cfg_log_level = gr.Dropdown(
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            value=initial_config.get("log_level", "INFO"),
                            label="Log Level",
                            scale=1,
                            info="Logging verbosity",
                        )
                    with gr.Row():
                        cfg_cors_origins = gr.Textbox(
                            value=initial_config.get(
                                "cors_origins", "http://localhost:3000,http://localhost:8080"
                            ),
                            label="CORS Origins (comma-separated)",
                            info="Allowed origins for cross-origin requests",
                        )

                # === GRAPHRAG SECTION ===
                with gr.Accordion("🕸️ GraphRAG", open=False):
                    gr.Markdown(
                        "*Enable knowledge graph extraction for relationship-aware queries. Best for protocols and structured data.*",
                        elem_classes=["help-text"],
                    )
                    with gr.Row():
                        cfg_enable_graph = gr.Checkbox(
                            value=initial_config.get("enable_graph_rag", True),
                            label="Enable GraphRAG",
                            info="Extract entities and relationships from documents",
                        )
                        cfg_graph_store = gr.Dropdown(
                            choices=["memory", "neo4j"],
                            value=initial_config.get("graph_store_backend", "memory"),
                            label="Graph Backend",
                            info="memory=in-memory (fast), neo4j=persistent database",
                        )
                    with gr.Row():
                        cfg_neo4j_uri = gr.Textbox(
                            value=initial_config.get("neo4j_uri", "bolt://localhost:7687"),
                            label="Neo4j URI",
                            info="Connection string for Neo4j database",
                        )
                        cfg_neo4j_user = gr.Textbox(
                            value=initial_config.get("neo4j_user", "neo4j"),
                            label="Neo4j User",
                            info="Neo4j username",
                        )
                        cfg_neo4j_database = gr.Textbox(
                            value=initial_config.get("neo4j_database", "neo4j"),
                            label="Neo4j Database",
                            info="Neo4j database name",
                        )

                # === QUERY & EXTRACTION SECTION ===
                with gr.Accordion("🔍 Query Routing & Extraction", open=False):
                    gr.Markdown(
                        "*Configure how queries are routed and entities are extracted. LLM modes require Ollama.*",
                        elem_classes=["help-text"],
                    )
                    with gr.Row():
                        cfg_router_mode = gr.Dropdown(
                            choices=["pattern", "llm"],
                            value=initial_config.get("router_mode", "pattern"),
                            label="Router Mode",
                            info="pattern=fast regex, llm=intelligent routing (needs Ollama)",
                        )
                        cfg_default_query_strategy = gr.Dropdown(
                            choices=["vector", "graph", "hybrid"],
                            value=initial_config.get("default_query_strategy", "vector"),
                            label="Default Strategy",
                            info="Fallback when router can't decide",
                        )
                        cfg_default_top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=initial_config.get("default_top_k", 5),
                            step=1,
                            label="Default Top K",
                            info="Default number of results to return",
                        )
                        cfg_entity_mode = gr.Dropdown(
                            choices=["rule_based", "llm"],
                            value=initial_config.get("entity_extraction_mode", "rule_based"),
                            label="Extraction Mode",
                            info="rule_based=fast, llm=better quality (needs Ollama)",
                        )
                        cfg_entity_domain = gr.Dropdown(
                            choices=["general", "mavlink"],
                            value=initial_config.get("entity_extraction_domain", "general"),
                            label="Domain",
                            info="Specialized extraction rules",
                        )
                        cfg_ollama_model = gr.Dropdown(
                            choices=["llama3.2", "llama3.1", "mistral", "gemma2", "phi3"],
                            value=initial_config.get("ollama_model", "llama3.2"),
                            label="Ollama Model",
                            allow_custom_value=True,
                            info="Model for LLM-based routing/extraction",
                        )

                # Action buttons and status at bottom
                gr.Markdown("---")
                with gr.Row():
                    load_config_btn = gr.Button(
                        "🔄 Reload from config.yaml", size="sm", variant="secondary"
                    )
                    save_config_btn = gr.Button(
                        "💾 Save to config.yaml", size="sm", variant="secondary"
                    )
                    apply_restart_btn = gr.Button("⚡ Save & Restart Service", variant="primary")

                config_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    placeholder="Make changes above, then click Save or Save & Restart",
                )

                # Config components list (order must match get_config_display return order)
                config_components = [
                    # Row 1: Embedding
                    cfg_embedding_model,
                    cfg_device,
                    cfg_embedding_batch_size,
                    # Row 2: Document Processing
                    cfg_chunk_size,
                    cfg_chunk_overlap,
                    cfg_pdf_strategy,
                    # Row 3: Vector Store
                    cfg_vector_store,
                    cfg_faiss_index_dir,
                    cfg_chroma_persist_dir,
                    cfg_default_collection,
                    # Row 4: API
                    cfg_host,
                    cfg_port,
                    cfg_api_prefix,
                    cfg_cors_origins,
                    cfg_log_level,
                    # Row 5: GraphRAG
                    cfg_enable_graph,
                    cfg_graph_store,
                    cfg_neo4j_uri,
                    cfg_neo4j_user,
                    cfg_neo4j_database,
                    # Row 6: Query & Extraction
                    cfg_router_mode,
                    cfg_default_query_strategy,
                    cfg_default_top_k,
                    cfg_entity_mode,
                    cfg_entity_domain,
                    cfg_ollama_model,
                ]

                def collect_config(*values: Any) -> dict[str, Any]:
                    """Collect config values into a dictionary."""
                    keys = [
                        # Embedding
                        "embedding_model",
                        "device",
                        "embedding_batch_size",
                        # Document Processing
                        "chunk_size",
                        "chunk_overlap",
                        "pdf_strategy",
                        # Vector Store
                        "vector_store_backend",
                        "faiss_index_dir",
                        "chroma_persist_dir",
                        "default_collection",
                        # API
                        "host",
                        "port",
                        "api_prefix",
                        "cors_origins",
                        "log_level",
                        # GraphRAG
                        "enable_graph_rag",
                        "graph_store_backend",
                        "neo4j_uri",
                        "neo4j_user",
                        "neo4j_database",
                        # Query & Extraction
                        "router_mode",
                        "default_query_strategy",
                        "default_top_k",
                        "entity_extraction_mode",
                        "entity_extraction_domain",
                        "ollama_model",
                    ]
                    return dict(zip(keys, values, strict=False))

                def save_and_report(*values: Any) -> str:
                    config = collect_config(*values)
                    return save_config_to_yaml(config)

                def apply_restart_and_report(*values: Any) -> str:
                    config = collect_config(*values)
                    return apply_and_restart(config)

                load_config_btn.click(
                    fn=get_config_display,
                    inputs=[],
                    outputs=config_components,
                )
                save_config_btn.click(
                    fn=save_and_report,
                    inputs=config_components,
                    outputs=[config_status],
                )
                apply_restart_btn.click(
                    fn=apply_restart_and_report,
                    inputs=config_components,
                    outputs=[config_status],
                )

            # Tab 5: System Info
            with gr.Tab("ℹ️ System"):
                gr.Markdown("*Click 'Load System Info' to see current system status*")
                system_info = gr.JSON(
                    label="System Information",
                    value={"status": "Click 'Load System Info' to refresh"},
                )

                with gr.Row():
                    load_sys_btn = gr.Button("ℹ️ Load System Info", variant="secondary", size="sm")
                    refresh_sys_btn = gr.Button("🔄 Refresh", size="sm")
                sys_status = gr.Textbox(label="Status", interactive=False, lines=2)

                load_sys_btn.click(
                    fn=safe_get_system_info,
                    inputs=[api_url_input],
                    outputs=[system_info, sys_status],
                )
                refresh_sys_btn.click(
                    fn=safe_get_system_info,
                    inputs=[api_url_input],
                    outputs=[system_info, sys_status],
                )

        gr.Markdown(
            """
            ---
            💡 **Tips:**
            - Use **auto** strategy for smart routing between vector and graph search
            - Larger **top_k** values provide more context but increase token usage
            - Configuration changes to embedding model/device require service restart
            """
        )

    return app  # type: ignore[no-any-return]


def main() -> None:
    """Launch the Gradio web interface."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Documentation Service UI")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"RAG API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"UI port (default: auto-detect from {DEFAULT_UI_PORT})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    args = parser.parse_args()

    # Find available port
    port = args.port or find_available_port(DEFAULT_UI_PORT)

    # Check API health
    is_healthy, status = check_api_health(args.api_url)
    if not is_healthy:
        logger.warning(status)
        print(f"\n⚠️  {status}")
        print("The UI will still launch, but you'll need to start the API service.\n")

    # Create and launch UI
    app = create_ui(api_url=args.api_url)

    print("\n🚀 Launching RAG Documentation Service UI")
    print(f"   Local URL: http://localhost:{port}")
    if args.share:
        print("   Creating public link...")

    app.launch(
        server_port=port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()

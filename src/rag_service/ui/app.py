"""Gradio web interface for the RAG Documentation Service.

Provides a simple drag-and-drop interface for document ingestion and
semantic search queries with hybrid vector/graph retrieval support.
"""

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_UI_PORT = 7860

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
        response = httpx.get(f"{api_url}/info", timeout=10.0)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_current_config() -> dict[str, Any]:
    """Load current configuration from environment and defaults."""
    from rag_service.config import get_settings

    try:
        settings = get_settings()
        return {
            "embedding_model": settings.embedding_model,
            "device": settings.device,
            "vector_store_backend": settings.vector_store_backend,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "enable_graph_rag": settings.enable_graph_rag,
            "graph_store_backend": settings.graph_store_backend,
            "router_mode": settings.router_mode,
            "entity_extraction_mode": settings.entity_extraction_mode,
            "log_level": settings.log_level,
        }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def save_config_to_env(config: dict[str, Any], env_path: Path = None) -> str:
    """Save configuration to .env file.

    Args:
        config: Configuration dictionary.
        env_path: Path to .env file.

    Returns:
        Status message.
    """
    if env_path is None:
        # Try common locations
        for path in [Path(".env"), Path("/app/.env"), Path.home() / ".rag_service.env"]:
            if path.parent.exists():
                env_path = path
                break
        if env_path is None:
            env_path = Path(".env")

    try:
        # Read existing .env content
        existing_vars = {}
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        existing_vars[key.strip()] = value.strip()

        # Update with new config (convert to env var format)
        env_mapping = {
            "embedding_model": "EMBEDDING_MODEL",
            "device": "DEVICE",
            "vector_store_backend": "VECTOR_STORE_BACKEND",
            "chunk_size": "CHUNK_SIZE",
            "chunk_overlap": "CHUNK_OVERLAP",
            "enable_graph_rag": "ENABLE_GRAPH_RAG",
            "graph_store_backend": "GRAPH_STORE_BACKEND",
            "router_mode": "ROUTER_MODE",
            "entity_extraction_mode": "ENTITY_EXTRACTION_MODE",
            "log_level": "LOG_LEVEL",
        }

        for key, env_key in env_mapping.items():
            if key in config:
                value = config[key]
                # Convert booleans to lowercase strings
                if isinstance(value, bool):
                    value = str(value).lower()
                existing_vars[env_key] = str(value)

        # Write back to .env
        with open(env_path, "w") as f:
            f.write("# RAG Service Configuration\n")
            f.write("# Generated by UI\n\n")
            for key, value in sorted(existing_vars.items()):
                f.write(f"{key}={value}\n")

        return f"✅ Configuration saved to {env_path}\n⚠️ Restart service to apply changes"

    except PermissionError:
        return f"❌ Permission denied: Cannot write to {env_path}"
    except Exception as e:
        return f"❌ Error saving config: {e}"


def apply_config_runtime(config: dict[str, Any]) -> str:
    """Apply configuration changes at runtime (where possible).

    Note: Some settings require restart to take effect.

    Args:
        config: Configuration dictionary.

    Returns:
        Status message.
    """
    try:
        # Set environment variables
        env_mapping = {
            "embedding_model": "EMBEDDING_MODEL",
            "device": "DEVICE",
            "vector_store_backend": "VECTOR_STORE_BACKEND",
            "chunk_size": "CHUNK_SIZE",
            "chunk_overlap": "CHUNK_OVERLAP",
            "enable_graph_rag": "ENABLE_GRAPH_RAG",
            "graph_store_backend": "GRAPH_STORE_BACKEND",
            "router_mode": "ROUTER_MODE",
            "entity_extraction_mode": "ENTITY_EXTRACTION_MODE",
            "log_level": "LOG_LEVEL",
        }

        for key, env_key in env_mapping.items():
            if key in config:
                value = config[key]
                if isinstance(value, bool):
                    value = str(value).lower()
                os.environ[env_key] = str(value)

        return "✅ Environment variables updated"

    except Exception as e:
        return f"❌ Error applying config: {e}"


def apply_and_restart(config: dict[str, Any]) -> str:
    """Apply configuration and restart the service.

    Saves config to .env, updates environment variables, then triggers
    a graceful shutdown. Docker will automatically restart the container.

    Args:
        config: Configuration dictionary.

    Returns:
        Status message (shown briefly before restart).
    """
    import signal
    import threading
    import time

    try:
        # Step 1: Save to .env for persistence
        save_result = save_config_to_env(config)
        if save_result.startswith("❌"):
            return save_result

        # Step 2: Update environment variables
        apply_config_runtime(config)

        # Step 3: Schedule graceful shutdown after response is sent
        def delayed_shutdown():
            time.sleep(1.5)  # Give time for response to be sent
            logger.info("Initiating service restart via SIGTERM...")
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=delayed_shutdown, daemon=True).start()

        return (
            "✅ Configuration saved!\n"
            "🔄 Service restarting in 2 seconds...\n"
            "⏳ Please wait ~30-60 seconds for reload.\n"
            "🔃 Page will auto-refresh when ready."
        )

    except Exception as e:
        logger.exception("Failed to apply and restart")
        return f"❌ Error: {e}"


def get_config_display() -> tuple:
    """Get current config values for UI display.

    Returns:
        Tuple of config values in order matching UI components.
    """
    config = load_current_config()
    return (
        config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        config.get("device", "auto"),
        config.get("vector_store_backend", "faiss"),
        config.get("chunk_size", 512),
        config.get("chunk_overlap", 50),
        config.get("enable_graph_rag", False),
        config.get("graph_store_backend", "memory"),
        config.get("router_mode", "pattern"),
        config.get("entity_extraction_mode", "rule_based"),
        config.get("log_level", "INFO"),
    )


def ingest_files(
    files: list,
    collection: str,
    api_url: str,
) -> str:
    """Ingest uploaded files into the RAG service.

    Args:
        files: List of uploaded file objects from Gradio.
        collection: Target collection name.
        api_url: Base URL of the RAG API service.

    Returns:
        Status message with results.
    """
    if not files:
        return "⚠️ No files selected"

    if not collection.strip():
        collection = "documents"

    results = []
    success_count = 0
    total_docs = 0

    for file in files:
        try:
            file_path = file.name if hasattr(file, "name") else str(file)

            response = httpx.post(
                f"{api_url}/api/v1/ingest/file",
                json={"path": file_path, "collection": collection},
                timeout=120.0,
            )

            if response.status_code == 200:
                data = response.json()
                docs = data.get("documents_processed", 0)
                total_docs += docs
                success_count += 1
                results.append(f"✅ {Path(file_path).name}: {docs} chunks")
            else:
                error = response.json().get("detail", "Unknown error")
                results.append(f"❌ {Path(file_path).name}: {error}")

        except Exception as e:
            results.append(f"❌ {Path(file_path).name}: {str(e)}")

    summary = f"\n{'─' * 40}\n📊 Summary: {success_count}/{len(files)} files, {total_docs} total chunks"
    return "\n".join(results) + summary


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


def list_collections(api_url: str) -> str:
    """List all available collections.

    Args:
        api_url: Base URL of the RAG API service.

    Returns:
        Formatted list of collections.
    """
    try:
        response = httpx.get(f"{api_url}/api/v1/collections", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            vector_cols = data.get("vector_collections", [])
            graph_cols = data.get("graph_collections", [])

            parts = ["📦 **Vector Collections:**"]
            if vector_cols:
                for col in vector_cols:
                    name = col.get("name", "unknown")
                    count = col.get("document_count", 0)
                    parts.append(f"  • {name}: {count} documents")
            else:
                parts.append("  (none)")

            if graph_cols:
                parts.append("\n🔗 **Graph Collections:**")
                for col in graph_cols:
                    name = col.get("name", "unknown")
                    nodes = col.get("total_nodes", 0)
                    rels = col.get("total_relationships", 0)
                    parts.append(f"  • {name}: {nodes} nodes, {rels} relationships")

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
    # Modern soft theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        theme=theme,
        title="RAG Documentation Service",
        css="""
        .gradio-container { max-width: 1200px !important; }
        .status-box { font-family: monospace; }
        
        /* Hide info text by default - hover tooltips */
        .config-field .info {
            opacity: 0;
            max-height: 0;
            overflow: hidden;
            transition: all 0.2s ease;
            font-size: 0.75rem;
            background: var(--neutral-100);
            border-radius: 4px;
            padding: 0;
            margin-top: 0;
        }
        .config-field:hover .info {
            opacity: 1;
            max-height: 100px;
            padding: 6px 8px;
            margin-top: 4px;
        }
        .config-field .label-wrap span {
            cursor: help;
        }
        """,
    ) as app:
        gr.Markdown(
            """
            # 📚 RAG Documentation Service
            
            Upload documents, build a knowledge base, and search with hybrid vector + graph retrieval.
            """
        )

        # API URL configuration (hidden by default)
        with gr.Accordion("⚙️ API Configuration", open=False):
            api_url_input = gr.Textbox(
                value=api_url,
                label="API URL",
                info="URL of the RAG API service",
            )
            health_btn = gr.Button("Check Connection", size="sm")
            health_status = gr.Textbox(label="Status", interactive=False)
            health_btn.click(
                fn=lambda url: check_api_health(url)[1],
                inputs=[api_url_input],
                outputs=[health_status],
            )

        with gr.Tabs():
            # Tab 1: Document Upload
            with gr.Tab("📁 Upload Documents"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Drag & Drop Files")
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".md", ".txt", ".xml", ".html", ".docx", ".py", ".js", ".json", ".yaml"],
                            label="Drop files here or click to browse",
                        )
                        upload_collection = gr.Textbox(
                            value="documents",
                            label="Collection Name",
                            info="Name for organizing your documents",
                        )
                        upload_btn = gr.Button("📤 Upload & Index", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### Or Index a Folder")
                        dir_path = gr.Textbox(
                            label="Directory Path",
                            placeholder="/path/to/your/documents",
                            info="Full path to document folder",
                        )
                        dir_collection = gr.Textbox(
                            value="documents",
                            label="Collection Name",
                        )
                        dir_recursive = gr.Checkbox(
                            value=True,
                            label="Include subdirectories",
                        )
                        dir_btn = gr.Button("📂 Index Folder", variant="primary")

                upload_output = gr.Textbox(
                    label="Results",
                    lines=10,
                    interactive=False,
                    elem_classes=["status-box"],
                )

                upload_btn.click(
                    fn=ingest_files,
                    inputs=[file_upload, upload_collection, api_url_input],
                    outputs=[upload_output],
                )
                dir_btn.click(
                    fn=ingest_directory,
                    inputs=[dir_path, dir_collection, dir_recursive, api_url_input],
                    outputs=[upload_output],
                )

            # Tab 2: Search
            with gr.Tab("🔍 Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Question",
                            placeholder="What would you like to know?",
                            lines=2,
                        )
                        with gr.Row():
                            search_collection = gr.Textbox(
                                value="documents",
                                label="Collection",
                                scale=2,
                            )
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Results",
                                scale=1,
                            )
                            strategy = gr.Dropdown(
                                choices=["auto", "vector", "graph", "hybrid"],
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

            # Tab 3: Collections
            with gr.Tab("📦 Collections"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                collections_display = gr.Markdown()

                with gr.Row():
                    delete_name = gr.Textbox(
                        label="Collection to Delete",
                        placeholder="Enter collection name",
                        scale=3,
                    )
                    delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                delete_output = gr.Textbox(label="Status", interactive=False)

                refresh_btn.click(
                    fn=list_collections,
                    inputs=[api_url_input],
                    outputs=[collections_display],
                )
                delete_btn.click(
                    fn=delete_collection,
                    inputs=[delete_name, api_url_input],
                    outputs=[delete_output],
                )

                # Load collections on tab open
                app.load(
                    fn=list_collections,
                    inputs=[api_url_input],
                    outputs=[collections_display],
                )

            # Tab 4: Configuration
            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("### Service Configuration")

                # Get initial config values
                initial_config = load_current_config()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Embedding & Processing**")
                        cfg_embedding_model = gr.Dropdown(
                            choices=CONFIG_SCHEMA["embedding_model"]["choices"],
                            value=initial_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                            label="Embedding Model ⓘ",
                            info=CONFIG_SCHEMA["embedding_model"]["tooltip"],
                            allow_custom_value=True,
                            elem_classes=["config-field"],
                        )
                        cfg_device = gr.Dropdown(
                            choices=CONFIG_SCHEMA["device"]["choices"],
                            value=initial_config.get("device", "auto"),
                            label="Device ⓘ",
                            info=CONFIG_SCHEMA["device"]["tooltip"],
                            elem_classes=["config-field"],
                        )
                        with gr.Row():
                            cfg_chunk_size = gr.Slider(
                                minimum=100,
                                maximum=4096,
                                value=initial_config.get("chunk_size", 512),
                                step=50,
                                label="Chunk Size ⓘ",
                                info=CONFIG_SCHEMA["chunk_size"]["tooltip"],
                                elem_classes=["config-field"],
                            )
                            cfg_chunk_overlap = gr.Slider(
                                minimum=0,
                                maximum=500,
                                value=initial_config.get("chunk_overlap", 50),
                                step=10,
                                label="Overlap ⓘ",
                                info=CONFIG_SCHEMA["chunk_overlap"]["tooltip"],
                                elem_classes=["config-field"],
                            )

                    with gr.Column():
                        gr.Markdown("**Storage & GraphRAG**")
                        cfg_vector_store = gr.Dropdown(
                            choices=CONFIG_SCHEMA["vector_store_backend"]["choices"],
                            value=initial_config.get("vector_store_backend", "faiss"),
                            label="Vector Store ⓘ",
                            info=CONFIG_SCHEMA["vector_store_backend"]["tooltip"],
                            elem_classes=["config-field"],
                        )
                        cfg_enable_graph = gr.Checkbox(
                            value=initial_config.get("enable_graph_rag", False),
                            label="Enable GraphRAG ⓘ",
                            info=CONFIG_SCHEMA["enable_graph_rag"]["tooltip"],
                            elem_classes=["config-field"],
                        )
                        with gr.Row():
                            cfg_graph_store = gr.Dropdown(
                                choices=CONFIG_SCHEMA["graph_store_backend"]["choices"],
                                value=initial_config.get("graph_store_backend", "memory"),
                                label="Graph Store ⓘ",
                                info=CONFIG_SCHEMA["graph_store_backend"]["tooltip"],
                                elem_classes=["config-field"],
                            )
                            cfg_router_mode = gr.Dropdown(
                                choices=CONFIG_SCHEMA["router_mode"]["choices"],
                                value=initial_config.get("router_mode", "pattern"),
                                label="Router ⓘ",
                                info=CONFIG_SCHEMA["router_mode"]["tooltip"],
                                elem_classes=["config-field"],
                            )
                        with gr.Row():
                            cfg_entity_mode = gr.Dropdown(
                                choices=CONFIG_SCHEMA["entity_extraction_mode"]["choices"],
                                value=initial_config.get("entity_extraction_mode", "rule_based"),
                                label="Extraction ⓘ",
                                info=CONFIG_SCHEMA["entity_extraction_mode"]["tooltip"],
                                elem_classes=["config-field"],
                            )
                            cfg_log_level = gr.Dropdown(
                                choices=CONFIG_SCHEMA["log_level"]["choices"],
                                value=initial_config.get("log_level", "INFO"),
                                label="Log Level ⓘ",
                                info=CONFIG_SCHEMA["log_level"]["tooltip"],
                                elem_classes=["config-field"],
                            )

                with gr.Row():
                    load_config_btn = gr.Button("🔄 Reload", size="sm", variant="secondary")
                    save_env_btn = gr.Button("💾 Save", size="sm", variant="secondary")
                    apply_restart_btn = gr.Button("⚡ Apply & Restart", variant="primary")

                config_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    placeholder="Changes require Apply & Restart to take effect",
                )

                # Config components list for easy reference
                config_components = [
                    cfg_embedding_model,
                    cfg_device,
                    cfg_vector_store,
                    cfg_chunk_size,
                    cfg_chunk_overlap,
                    cfg_enable_graph,
                    cfg_graph_store,
                    cfg_router_mode,
                    cfg_entity_mode,
                    cfg_log_level,
                ]

                def collect_config(*values):
                    """Collect config values into a dictionary."""
                    keys = [
                        "embedding_model",
                        "device",
                        "vector_store_backend",
                        "chunk_size",
                        "chunk_overlap",
                        "enable_graph_rag",
                        "graph_store_backend",
                        "router_mode",
                        "entity_extraction_mode",
                        "log_level",
                    ]
                    return dict(zip(keys, values))

                def save_and_report(*values):
                    config = collect_config(*values)
                    return save_config_to_env(config)

                def apply_restart_and_report(*values):
                    config = collect_config(*values)
                    return apply_and_restart(config)

                load_config_btn.click(
                    fn=get_config_display,
                    inputs=[],
                    outputs=config_components,
                )
                save_env_btn.click(
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
                system_info = gr.JSON(label="System Information")
                refresh_sys_btn = gr.Button("🔄 Refresh")
                refresh_sys_btn.click(
                    fn=get_system_info,
                    inputs=[api_url_input],
                    outputs=[system_info],
                )
                app.load(
                    fn=get_system_info,
                    inputs=[api_url_input],
                    outputs=[system_info],
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

    return app


def main():
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

    print(f"\n🚀 Launching RAG Documentation Service UI")
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


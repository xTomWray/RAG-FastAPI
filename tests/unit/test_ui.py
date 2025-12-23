"""Unit tests for the RAG Documentation Service UI.

Tests verify that the Gradio UI components are correctly configured
and functional without requiring the full API to be running.
"""


class TestUIConfiguration:
    """Test UI configuration and constants."""

    def test_default_api_url(self):
        """Test that default API URL is configured."""
        from rag_service.ui.app import DEFAULT_API_URL

        assert DEFAULT_API_URL == "http://localhost:8080"

    def test_default_ui_port(self):
        """Test that default UI port is configured."""
        from rag_service.ui.app import DEFAULT_UI_PORT

        assert DEFAULT_UI_PORT == 7860

    def test_config_schema_exists(self):
        """Test that configuration schema is defined."""
        from rag_service.ui.app import CONFIG_SCHEMA

        assert isinstance(CONFIG_SCHEMA, dict)
        assert "embedding_model" in CONFIG_SCHEMA
        assert "device" in CONFIG_SCHEMA
        assert "vector_store_backend" in CONFIG_SCHEMA
        assert "chunk_size" in CONFIG_SCHEMA
        assert "chunk_overlap" in CONFIG_SCHEMA
        assert "enable_graph_rag" in CONFIG_SCHEMA
        assert "graph_store_backend" in CONFIG_SCHEMA
        assert "router_mode" in CONFIG_SCHEMA
        assert "entity_extraction_mode" in CONFIG_SCHEMA
        assert "log_level" in CONFIG_SCHEMA

    def test_config_schema_has_tooltips(self):
        """Test that all config options have tooltips."""
        from rag_service.ui.app import CONFIG_SCHEMA

        for key, value in CONFIG_SCHEMA.items():
            assert "tooltip" in value, f"Config option '{key}' missing tooltip"
            assert len(value["tooltip"]) > 0, f"Config option '{key}' has empty tooltip"


class TestUIHelperFunctions:
    """Test UI helper functions."""

    def test_escape_html_basic(self):
        """Test HTML escaping for basic characters."""
        from rag_service.ui.app import escape_html

        assert escape_html("Hello & World") == "Hello &amp; World"
        assert escape_html("<script>") == "&lt;script&gt;"
        assert escape_html('"quoted"') == "&quot;quoted&quot;"

    def test_escape_html_combined(self):
        """Test HTML escaping with multiple special characters."""
        from rag_service.ui.app import escape_html

        result = escape_html('<a href="test">Link & Text</a>')
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result

    def test_escape_html_safe_text(self):
        """Test that safe text is unchanged."""
        from rag_service.ui.app import escape_html

        safe_text = "Normal text without special characters"
        assert escape_html(safe_text) == safe_text

    def test_find_available_port(self):
        """Test port finding function."""
        from rag_service.ui.app import find_available_port

        # Should find a port starting from 7860
        port = find_available_port(7860, max_attempts=100)
        assert isinstance(port, int)
        assert port >= 7860
        assert port < 7960

    def test_check_api_health_returns_tuple(self):
        """Test that check_api_health returns correct format."""
        from rag_service.ui.app import check_api_health

        result = check_api_health("http://localhost:9999")  # Non-existent
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_healthy, message = result
        assert isinstance(is_healthy, bool)
        assert isinstance(message, str)

    def test_check_api_health_failure(self):
        """Test check_api_health with unreachable server."""
        from rag_service.ui.app import check_api_health

        is_healthy, message = check_api_health("http://localhost:9999")
        assert is_healthy is False
        assert "❌" in message or "Cannot connect" in message


class TestUICreation:
    """Test UI creation and structure."""

    def test_create_ui_returns_blocks(self):
        """Test that create_ui returns a Gradio Blocks instance."""
        import gradio as gr

        from rag_service.ui.app import create_ui

        ui = create_ui(api_url="http://localhost:8080")
        assert isinstance(ui, gr.Blocks)

    def test_create_ui_with_custom_url(self):
        """Test create_ui accepts custom API URL."""
        from rag_service.ui.app import create_ui

        # Should not raise any errors
        ui = create_ui(api_url="http://custom-host:9000")
        assert ui is not None

    def test_create_ui_has_title(self):
        """Test that created UI has correct title."""
        from rag_service.ui.app import create_ui

        ui = create_ui()
        assert ui.title == "RAG Documentation Service"


class TestConfigFunctions:
    """Test configuration loading and saving functions."""

    def test_load_current_config_returns_dict(self):
        """Test that load_current_config returns a dictionary."""
        from rag_service.ui.app import load_current_config

        config = load_current_config()
        assert isinstance(config, dict)

    def test_load_current_config_has_required_keys(self):
        """Test that loaded config has expected keys."""
        from rag_service.ui.app import load_current_config

        config = load_current_config()
        expected_keys = [
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
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"

    def test_get_config_display_returns_tuple(self):
        """Test that get_config_display returns correct format."""
        from rag_service.ui.app import get_config_display

        result = get_config_display()
        assert isinstance(result, tuple)
        # Config display returns all config values (may vary based on settings)
        assert len(result) >= 25  # At least 25 config values


class TestIngestFunctions:
    """Test document ingestion helper functions."""

    def test_ingest_files_empty_list(self):
        """Test ingest_files with empty file list."""
        from rag_service.ui.app import ingest_files

        result = ingest_files([], "test_collection", "http://localhost:8080")
        # Function returns a tuple: (status_message, file_list, markdown_message)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert "No files selected" in result[0]  # Check the status message

    def test_ingest_directory_empty_path(self):
        """Test ingest_directory with empty path."""
        from rag_service.ui.app import ingest_directory

        result = ingest_directory("", "test_collection", True, "http://localhost:8080")
        assert "Please enter a directory path" in result

    def test_ingest_directory_whitespace_path(self):
        """Test ingest_directory with whitespace-only path."""
        from rag_service.ui.app import ingest_directory

        result = ingest_directory("   ", "test_collection", True, "http://localhost:8080")
        assert "Please enter a directory path" in result


class TestQueryFunctions:
    """Test query helper functions."""

    def test_query_documents_empty_question(self):
        """Test query_documents with empty question."""
        from rag_service.ui.app import query_documents

        results, sources, metadata = query_documents(
            "", "test_collection", 5, "auto", "http://localhost:8080"
        )
        assert "Please enter a question" in results
        assert sources == ""
        assert metadata == "{}"

    def test_query_documents_whitespace_question(self):
        """Test query_documents with whitespace-only question."""
        from rag_service.ui.app import query_documents

        results, sources, metadata = query_documents(
            "   ", "test_collection", 5, "auto", "http://localhost:8080"
        )
        assert "Please enter a question" in results


class TestCollectionFunctions:
    """Test collection management functions."""

    def test_delete_collection_empty_name(self):
        """Test delete_collection with empty name."""
        from rag_service.ui.app import delete_collection

        result = delete_collection("", "http://localhost:8080")
        assert "Please enter a collection name" in result

    def test_delete_collection_whitespace_name(self):
        """Test delete_collection with whitespace-only name."""
        from rag_service.ui.app import delete_collection

        result = delete_collection("   ", "http://localhost:8080")
        assert "Please enter a collection name" in result


class TestSystemInfoFunctions:
    """Test system info functions."""

    def test_get_system_info_returns_dict(self):
        """Test that get_system_info returns a dictionary."""
        from rag_service.ui.app import get_system_info

        # With unreachable server, should return error dict
        result = get_system_info("http://localhost:9999")
        assert isinstance(result, dict)


class TestUIIntegration:
    """Integration tests for the UI module."""

    def test_module_imports_successfully(self):
        """Test that the UI module can be imported without errors."""
        import rag_service.ui.app

        assert hasattr(rag_service.ui.app, "create_ui")
        assert hasattr(rag_service.ui.app, "main")

    def test_gradio_dependency_available(self):
        """Test that Gradio is installed and importable."""
        import gradio as gr

        assert hasattr(gr, "Blocks")
        assert hasattr(gr, "Textbox")
        assert hasattr(gr, "Button")
        assert hasattr(gr, "Tab")

    def test_httpx_dependency_available(self):
        """Test that httpx is installed and importable."""
        import httpx

        assert hasattr(httpx, "get")
        assert hasattr(httpx, "post")

"""Unit tests for the document chunker."""

from pathlib import Path

import pytest

from rag_service.core.chunker import DocumentChunker, create_chunker
from rag_service.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_chunk_short_text(self, chunker: DocumentChunker) -> None:
        """Test that short text is not chunked."""
        text = "This is a short text."
        chunks = chunker._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self, chunker: DocumentChunker) -> None:
        """Test that long text is chunked correctly."""
        # Create text longer than chunk_size (100)
        text = "Word " * 50  # 250 characters
        chunks = chunker._chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_overlap(self) -> None:
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 30 + " " + "B" * 30 + " " + "C" * 30
        chunks = chunker._chunk_text(text)

        # With overlap, content should appear in multiple chunks
        assert len(chunks) >= 2

    def test_empty_text(self, chunker: DocumentChunker) -> None:
        """Test handling of empty text."""
        chunks = chunker._chunk_text("")
        assert len(chunks) == 0

        chunks = chunker._chunk_text("   ")
        assert len(chunks) == 0

    def test_process_text_file(
        self,
        chunker: DocumentChunker,
        sample_text_file: Path,
    ) -> None:
        """Test processing a text file."""
        documents = chunker.process_file(sample_text_file)

        assert len(documents) > 0
        assert all(doc.text for doc in documents)
        assert all(doc.metadata.get("source") == str(sample_text_file) for doc in documents)
        assert all(doc.metadata.get("filename") == "sample.txt" for doc in documents)

    def test_process_nonexistent_file(self, chunker: DocumentChunker) -> None:
        """Test error handling for nonexistent file."""
        with pytest.raises(DocumentProcessingError, match="File not found"):
            chunker.process_file(Path("/nonexistent/file.txt"))

    def test_process_unsupported_file(
        self,
        chunker: DocumentChunker,
        temp_dir: Path,
    ) -> None:
        """Test error handling for unsupported file types."""
        unsupported_file = temp_dir / "file.xyz"
        unsupported_file.write_text("content")

        with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
            chunker.process_file(unsupported_file)

    def test_process_directory(
        self,
        chunker: DocumentChunker,
        temp_dir: Path,
    ) -> None:
        """Test processing a directory of files."""
        # Create test files
        (temp_dir / "file1.txt").write_text("Content of file 1")
        (temp_dir / "file2.md").write_text("# Markdown file\n\nContent here")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file3.txt").write_text("Nested content")

        documents = chunker.process_directory(temp_dir, recursive=True)

        # Should find all 3 files
        sources = {doc.metadata.get("source") for doc in documents}
        assert len(sources) == 3

    def test_process_directory_non_recursive(
        self,
        chunker: DocumentChunker,
        temp_dir: Path,
    ) -> None:
        """Test non-recursive directory processing."""
        (temp_dir / "file1.txt").write_text("Content")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("Nested")

        documents = chunker.process_directory(temp_dir, recursive=False)

        sources = {doc.metadata.get("source") for doc in documents}
        assert len(sources) == 1

    def test_generate_doc_id(self, chunker: DocumentChunker) -> None:
        """Test document ID generation is deterministic."""
        path = Path("/test/file.txt")
        id1 = chunker._generate_doc_id(path, 0)
        id2 = chunker._generate_doc_id(path, 0)
        id3 = chunker._generate_doc_id(path, 1)

        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16

    def test_create_chunker_factory(self) -> None:
        """Test the factory function."""
        chunker = create_chunker(chunk_size=200, chunk_overlap=30)
        assert chunker._chunk_size == 200
        assert chunker._chunk_overlap == 30

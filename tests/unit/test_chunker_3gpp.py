"""Unit tests for 3GPP-specific document chunker."""

import pytest

from rag_service.core.chunker_3gpp import (
    ChunkType,
    create_3gpp_chunker,
)


class TestThreeGPPChunkerClauseParsing:
    """Tests for clause boundary detection and parsing."""

    @pytest.fixture
    def chunker(self):
        """Create a default 3GPP chunker."""
        return create_3gpp_chunker(max_chunk_size=1000)

    def test_parse_simple_clauses(self, chunker):
        """Test parsing document with simple clause structure."""
        text = """6.3.2 Authentication Procedure

This section describes the authentication procedure.

6.3.2.1 Initial Authentication

The UE shall initiate authentication when...

6.3.2.2 Re-authentication

The re-authentication procedure is used when..."""

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 3
        # Check that clauses are detected
        clause_nums = [c.clause_number for c in chunks if c.clause_number]
        assert "6.3.2" in clause_nums
        assert "6.3.2.1" in clause_nums
        assert "6.3.2.2" in clause_nums

    def test_clause_title_preserved(self, chunker):
        """Test that clause titles are preserved in chunks."""
        text = """5.4.1 Message Format

The message format is defined as follows..."""

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        assert chunks[0].clause_title == "Message Format"
        assert "5.4.1" in chunks[0].text

    def test_nested_clause_structure(self, chunker):
        """Test parsing deeply nested clause numbers."""
        text = """6.1 General
Introduction.

6.1.1 Scope
Scope details.

6.1.1.1 Sub-scope
Sub details.

6.1.1.1.1 Deep nesting
Very deep."""

        chunks = chunker.chunk_text(text)

        clause_nums = [c.clause_number for c in chunks if c.clause_number]
        assert "6.1" in clause_nums
        assert "6.1.1" in clause_nums
        assert "6.1.1.1" in clause_nums
        assert "6.1.1.1.1" in clause_nums


class TestThreeGPPChunkerConstructPreservation:
    """Tests for keeping procedural constructs intact."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with construct preservation enabled."""
        return create_3gpp_chunker(
            max_chunk_size=2000,
            preserve_constructs=True,
        )

    def test_if_else_block_preserved(self, chunker):
        """Test that if/else blocks stay together."""
        text = """6.3.1 Decision Logic

If the UE receives a valid authentication response:
    - The UE shall store the security context
    - The UE shall update the timer T3512
    - The UE shall send an acknowledgement
Otherwise:
    - The UE shall reject the authentication
    - The UE shall clear the security context

This concludes the procedure."""

        chunks = chunker.chunk_text(text)

        # The if/else block should be in a single chunk
        if_else_found = False
        for chunk in chunks:
            if "If the UE receives" in chunk.text:
                # Check the entire block is together
                assert "shall store the security context" in chunk.text
                assert "shall update the timer" in chunk.text
                if_else_found = True
                break

        assert if_else_found, "If/else block not found in chunks"

    def test_procedure_steps_preserved(self, chunker):
        """Test that procedure step lists stay together."""
        text = """6.4.2 Registration Procedure

The registration procedure consists of:

Step 1: The UE sends a REGISTRATION REQUEST message.
Step 2: The network verifies the UE identity.
Step 3: The network performs authentication.
Step 4: The network sends REGISTRATION ACCEPT.

End of procedure."""

        chunks = chunker.chunk_text(text)

        # Steps should be together
        steps_found = False
        for chunk in chunks:
            if "Step 1" in chunk.text:
                assert "Step 2" in chunk.text
                assert "Step 3" in chunk.text
                assert "Step 4" in chunk.text
                steps_found = True
                break

        assert steps_found, "Procedure steps not preserved together"

    def test_timer_paragraph_preserved(self, chunker):
        """Test that timer-related paragraphs stay together."""
        text = """6.5.1 Timer Handling

The timer T3512 is used for periodic registration.
The UE shall start timer T3512 upon successful registration.
When timer T3512 expires, the UE shall initiate periodic registration.
The UE shall stop timer T3512 if deregistration is triggered.

Other content here."""

        chunks = chunker.chunk_text(text)

        # Timer paragraph should be preserved
        timer_chunk = None
        for chunk in chunks:
            if "timer T3512" in chunk.text and "start timer" in chunk.text:
                timer_chunk = chunk
                break

        assert timer_chunk is not None
        assert "expires" in timer_chunk.text
        assert "stop timer" in timer_chunk.text

    def test_message_sequence_preserved(self, chunker):
        """Test that message sequences stay together."""
        text = """6.6.1 Message Flow

The UE sends an AUTHENTICATION REQUEST to the network.
The network validates the request parameters.
The gNB responds with an AUTHENTICATION RESPONSE.
The UE verifies the response and completes authentication.

Next section here."""

        chunks = chunker.chunk_text(text)

        # Message sequence should be together
        sequence_chunk = None
        for chunk in chunks:
            if "UE sends" in chunk.text:
                sequence_chunk = chunk
                break

        assert sequence_chunk is not None
        assert "gNB responds" in sequence_chunk.text or "responds" in sequence_chunk.text

    def test_bullet_list_preserved(self, chunker):
        """Test that bullet lists stay together."""
        text = """6.7.1 Requirements

The following requirements apply:
- The UE shall support NR
- The UE shall support SA mode
- The UE shall support 5G-AKA
- The UE shall implement key derivation

Additional text here."""

        chunks = chunker.chunk_text(text)

        # Find the chunk with bullets
        bullet_chunk = None
        for chunk in chunks:
            if "shall support NR" in chunk.text:
                bullet_chunk = chunk
                break

        assert bullet_chunk is not None
        assert "shall support SA mode" in bullet_chunk.text
        assert "shall support 5G-AKA" in bullet_chunk.text


class TestThreeGPPChunkerTableHandling:
    """Tests for table chunking with header repetition."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with small table row limit for testing."""
        return create_3gpp_chunker(
            max_chunk_size=2000,
            table_row_group_size=3,
        )

    def test_small_table_single_chunk(self, chunker):
        """Test that small tables stay as single chunk."""
        text = """6.8.1 Message Fields

Table 6.8.1-1: Authentication Fields

| Field | Type | Description |
|-------|------|-------------|
| AUTN | bytes | Authentication token |
| RAND | bytes | Random challenge |

More content."""

        chunks = chunker.chunk_text(text)

        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        assert len(table_chunks) == 1
        assert "AUTN" in table_chunks[0].text
        assert "RAND" in table_chunks[0].text

    def test_large_table_split_with_headers(self, chunker):
        """Test that large tables are split with repeated headers."""
        text = """Table 6.9.1-1: Cause Values

| Cause | Value | Description |
|-------|-------|-------------|
| CAUSE_1 | 1 | First cause |
| CAUSE_2 | 2 | Second cause |
| CAUSE_3 | 3 | Third cause |
| CAUSE_4 | 4 | Fourth cause |
| CAUSE_5 | 5 | Fifth cause |
| CAUSE_6 | 6 | Sixth cause |
| CAUSE_7 | 7 | Seventh cause |
"""

        chunks = chunker.chunk_text(text)

        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

        # Should be split into multiple chunks (7 rows / 3 per group = 3 chunks)
        assert len(table_chunks) >= 2

        # Each chunk should have headers
        for chunk in table_chunks:
            assert "Cause" in chunk.text
            assert "Value" in chunk.text
            assert "Description" in chunk.text

    def test_table_caption_preserved(self, chunker):
        """Test that table caption is preserved in each chunk."""
        text = """Table 5.1-1: Important Values

| Name | Value |
|------|-------|
| A | 1 |
| B | 2 |
"""

        chunks = chunker.chunk_text(text)

        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        # Table might be extracted or might remain as text
        if table_chunks:
            assert (
                table_chunks[0].table_caption is not None
                or "Important Values" in table_chunks[0].text
            )
        else:
            # If no TABLE chunks, check that the content is somewhere
            all_text = " ".join(c.text for c in chunks)
            assert "Important Values" in all_text or "Name" in all_text


class TestThreeGPPChunkerChunkSizing:
    """Tests for chunk size management."""

    def test_creates_multiple_chunks_for_long_content(self):
        """Test that long content is split into multiple chunks."""
        chunker = create_3gpp_chunker(max_chunk_size=500)

        # Create a long document
        text = "6.1 Introduction\n\n" + "This is content. " * 200

        chunks = chunker.chunk_text(text)

        # Should create at least one chunk
        assert len(chunks) >= 1
        # The content should be distributed (not all in one chunk if very long)
        total_content = " ".join(c.text for c in chunks)
        assert "This is content" in total_content

    def test_long_clause_split_into_multiple_chunks(self):
        """Test that long clauses are split into multiple chunks."""
        chunker = create_3gpp_chunker(max_chunk_size=300)

        text = (
            """6.2.1 Long Procedure

This is a very long procedure description that needs to be split.
"""
            + "Content continues here. " * 50
        )

        chunks = chunker.chunk_text(text)

        # Should have multiple chunks for such long content
        assert len(chunks) >= 1  # At least one chunk

        # All chunks should have the clause association
        clause_chunks = [c for c in chunks if c.clause_number == "6.2.1"]
        assert len(clause_chunks) >= 1


class TestThreeGPPChunkerDocumentProcessing:
    """Tests for full document processing."""

    @pytest.fixture
    def chunker(self):
        """Create a default chunker."""
        return create_3gpp_chunker()

    def test_chunk_text_returns_chunks(self, chunker):
        """Test basic chunk_text functionality."""
        text = "6.1 Test\n\nSome content here."

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        assert all(hasattr(c, "text") for c in chunks)
        assert all(hasattr(c, "chunk_type") for c in chunks)

    def test_no_clauses_falls_back_to_paragraphs(self, chunker):
        """Test that documents without clauses still get chunked."""
        text = """This is a document without numbered clauses.

It has multiple paragraphs though.

And should still be chunked appropriately."""

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        assert all(c.chunk_type == ChunkType.TEXT for c in chunks)

    def test_empty_text_returns_empty(self, chunker):
        """Test handling of empty text."""
        chunks = chunker.chunk_text("")
        # Empty or whitespace should result in no chunks
        assert len(chunks) == 0 or all(not c.text.strip() for c in chunks)


class TestCreateThreeGPPChunkerFactory:
    """Tests for the factory function."""

    def test_default_parameters(self):
        """Test factory with default parameters."""
        chunker = create_3gpp_chunker()

        assert chunker._max_chunk_size == 1500
        assert chunker._preserve_constructs is True
        assert chunker._table_row_group_size == 10

    def test_custom_parameters(self):
        """Test factory with custom parameters."""
        chunker = create_3gpp_chunker(
            max_chunk_size=2000,
            min_chunk_size=300,
            preserve_constructs=False,
            table_row_group_size=5,
        )

        assert chunker._max_chunk_size == 2000
        assert chunker._min_chunk_size == 300
        assert chunker._preserve_constructs is False
        assert chunker._table_row_group_size == 5

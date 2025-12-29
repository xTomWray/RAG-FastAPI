"""3GPP-specific document chunking with clause-aware splitting.

Implements specialized chunking rules for 3GPP specification documents:
1. Clause boundary chunking (e.g., 6.3.2.1, 5.3.1.2)
2. Construct preservation (if/else/when, procedure steps, timers, message sequences)
3. Table handling with header repetition

3GPP specifications have a very specific structure that generic chunkers miss:
- Numbered clauses form natural boundaries
- Procedural logic (if/then/else) should stay together
- Tables define IEs, causes, and conditions that should remain atomic
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rag_service.core.retriever import Document

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks in 3GPP documents."""

    CLAUSE = "clause"
    TABLE = "table"
    PROCEDURE = "procedure"
    TEXT = "text"


@dataclass
class ParsedClause:
    """A parsed clause from a 3GPP document."""

    number: str  # e.g., "6.3.2.1"
    title: str
    content: str
    level: int  # Depth of clause numbering
    start_pos: int
    end_pos: int
    children: list["ParsedClause"] = field(default_factory=list)


@dataclass
class ParsedTable:
    """A parsed table from a 3GPP document."""

    caption: str
    headers: list[str]
    rows: list[list[str]]
    start_pos: int
    end_pos: int


@dataclass
class ThreeGPPChunk:
    """A chunk extracted from a 3GPP document."""

    text: str
    chunk_type: ChunkType
    clause_number: str | None = None
    clause_title: str | None = None
    table_caption: str | None = None
    metadata: dict[str, str | int] = field(default_factory=dict)


class ThreeGPPChunker:
    """Chunker specialized for 3GPP specification documents.

    Recognizes and preserves:
    - Clause boundaries (numbered sections like 6.3.2.1)
    - Bullet list blocks (kept together)
    - Procedural constructs (if/else/when blocks, step lists)
    - Timer paragraphs (start/stop/expiry behavior)
    - Message sequences (UE sends... gNB responds...)
    - Tables with caption/header/row structure
    """

    # Regex patterns for 3GPP document elements
    CLAUSE_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)*)\s+([^\n]+)",
        re.MULTILINE,
    )

    # Patterns for constructs that should NOT be split
    CONSTRUCT_PATTERNS = {
        "if_else": re.compile(
            r"(?:^|\n)(?:If|When|Upon receipt)[^\n]*(?:\n(?!(?:\d+\.|\n\n))[^\n]*)*",
            re.IGNORECASE | re.MULTILINE,
        ),
        "procedure_steps": re.compile(
            r"(?:^|\n)Step\s*\d+[^\n]*(?:\n(?!Step\s*\d+)[^\n]*)*",
            re.IGNORECASE | re.MULTILINE,
        ),
        "timer_paragraph": re.compile(
            r"(?:^|\n)[^\n]*(?:start|stop|expir)[^\n]*timer[^\n]*(?:\n(?!\n\n)[^\n]*)*",
            re.IGNORECASE | re.MULTILINE,
        ),
        "message_sequence": re.compile(
            r"(?:^|\n)(?:The\s+)?(?:UE|gNB|eNB|AMF|SMF|NRF|MME)[^\n]*(?:sends?|receives?|responds?)[^\n]*(?:\n(?!\n\n)[^\n]*)*",
            re.IGNORECASE | re.MULTILINE,
        ),
    }

    # Pattern for bullet lists
    BULLET_LIST_PATTERN = re.compile(
        r"(?:^|\n)([-•]\s+[^\n]+(?:\n(?![-•]|\n\n)[^\n]*)*(?:\n[-•]\s+[^\n]+(?:\n(?![-•]|\n\n)[^\n]*)*)*)",
        re.MULTILINE,
    )

    # Table patterns
    TABLE_START_PATTERN = re.compile(
        r"(?:Table\s+[\d.]+[:\s]*[^\n]*|^\|[^\n]+\|$)",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        chunk_overlap: int = 100,
        preserve_constructs: bool = True,
        table_row_group_size: int = 10,
    ) -> None:
        """Initialize the 3GPP chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            min_chunk_size: Minimum chunk size (smaller chunks merged).
            chunk_overlap: Overlap between chunks when splitting.
            preserve_constructs: Keep if/else, procedures, etc. together.
            table_row_group_size: Max rows per table chunk before splitting.
        """
        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size
        self._chunk_overlap = chunk_overlap
        self._preserve_constructs = preserve_constructs
        self._table_row_group_size = table_row_group_size

    def chunk_text(self, text: str, _source: str = "") -> list[ThreeGPPChunk]:
        """Chunk 3GPP document text using clause-aware splitting.

        Args:
            text: The document text to chunk.
            _source: Source file path for metadata (unused, for API compatibility).

        Returns:
            List of ThreeGPPChunk objects.
        """
        chunks: list[ThreeGPPChunk] = []

        # Step 1: Extract and process tables separately
        text_without_tables, tables = self._extract_tables(text)

        # Step 2: Parse clause structure
        clauses = self._parse_clauses(text_without_tables)

        # Step 3: Process each clause
        if clauses:
            for clause in clauses:
                clause_chunks = self._chunk_clause(clause)
                chunks.extend(clause_chunks)
        else:
            # No clause structure found - use paragraph-based chunking
            chunks.extend(self._chunk_paragraphs(text_without_tables))

        # Step 4: Add table chunks
        for table in tables:
            table_chunks = self._chunk_table(table)
            chunks.extend(table_chunks)

        logger.debug(f"Created {len(chunks)} chunks from 3GPP document")
        return chunks

    def _extract_tables(self, text: str) -> tuple[str, list[ParsedTable]]:
        """Extract tables from text and return text without tables.

        Args:
            text: Document text.

        Returns:
            Tuple of (text_without_tables, list_of_tables).
        """
        tables: list[ParsedTable] = []

        # Find markdown-style tables
        table_pattern = re.compile(
            r"((?:Table\s+[\d.]+[:\s]*[^\n]*\n)?)"  # Optional caption
            r"(\|[^\n]+\|\n)"  # Header row
            r"(\|[-:| ]+\|\n)"  # Separator
            r"((?:\|[^\n]+\|\n)+)",  # Data rows
            re.MULTILINE,
        )

        text_without_tables = text
        offset = 0

        for match in table_pattern.finditer(text):
            caption = match.group(1).strip() if match.group(1) else ""
            header_line = match.group(2).strip()
            data_lines = match.group(4).strip()

            # Parse header
            headers = [cell.strip() for cell in header_line.strip("|").split("|") if cell.strip()]

            # Parse rows
            rows = []
            for line in data_lines.split("\n"):
                if line.strip():
                    row = [cell.strip() for cell in line.strip("|").split("|")]
                    rows.append(row)

            tables.append(
                ParsedTable(
                    caption=caption,
                    headers=headers,
                    rows=rows,
                    start_pos=match.start() - offset,
                    end_pos=match.end() - offset,
                )
            )

            # Remove table from text (replace with marker)
            marker = f"\n[TABLE: {caption}]\n" if caption else "\n[TABLE]\n"
            text_without_tables = (
                text_without_tables[: match.start() - offset]
                + marker
                + text_without_tables[match.end() - offset :]
            )
            offset += len(match.group(0)) - len(marker)

        return text_without_tables, tables

    def _parse_clauses(self, text: str) -> list[ParsedClause]:
        """Parse clause structure from 3GPP document.

        Args:
            text: Document text.

        Returns:
            List of top-level ParsedClause objects.
        """
        clauses: list[ParsedClause] = []
        matches = list(self.CLAUSE_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            number = match.group(1)
            title = match.group(2).strip()
            level = number.count(".") + 1

            # Determine content end
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Extract content (everything after the header until next clause)
            content_start = match.end()
            content = text[content_start:end_pos].strip()

            clause = ParsedClause(
                number=number,
                title=title,
                content=content,
                level=level,
                start_pos=match.start(),
                end_pos=end_pos,
            )
            clauses.append(clause)

        return clauses

    def _chunk_clause(self, clause: ParsedClause) -> list[ThreeGPPChunk]:
        """Chunk a single clause, preserving constructs.

        Args:
            clause: The parsed clause to chunk.

        Returns:
            List of chunks from this clause.
        """
        chunks: list[ThreeGPPChunk] = []

        # Create header text
        header = f"{clause.number} {clause.title}\n\n"
        content = clause.content

        # If content is small enough, return as single chunk
        if len(header) + len(content) <= self._max_chunk_size:
            chunks.append(
                ThreeGPPChunk(
                    text=header + content,
                    chunk_type=ChunkType.CLAUSE,
                    clause_number=clause.number,
                    clause_title=clause.title,
                )
            )
            return chunks

        # Split content while preserving constructs
        segments = self._split_preserving_constructs(content)

        current_chunk = header
        for segment in segments:
            # Check if adding this segment exceeds max size
            if len(current_chunk) + len(segment) > self._max_chunk_size:
                # Save current chunk if it has content
                if len(current_chunk) > len(header):
                    chunks.append(
                        ThreeGPPChunk(
                            text=current_chunk.strip(),
                            chunk_type=ChunkType.CLAUSE,
                            clause_number=clause.number,
                            clause_title=clause.title,
                        )
                    )
                # Start new chunk with header for context
                current_chunk = f"{clause.number} {clause.title} (cont.)\n\n" + segment
            else:
                current_chunk += segment

        # Don't forget the last chunk
        if current_chunk.strip() and len(current_chunk) > len(header):
            chunks.append(
                ThreeGPPChunk(
                    text=current_chunk.strip(),
                    chunk_type=ChunkType.CLAUSE,
                    clause_number=clause.number,
                    clause_title=clause.title,
                )
            )

        return chunks

    def _split_preserving_constructs(self, text: str) -> list[str]:
        """Split text into segments, keeping constructs intact.

        Args:
            text: Text to split.

        Returns:
            List of text segments.
        """
        if not self._preserve_constructs:
            # Simple paragraph split
            return self._split_paragraphs(text)

        segments: list[str] = []
        protected_regions: list[tuple[int, int, str]] = []  # (start, end, matched_text)

        # Find all protected constructs
        for _name, pattern in self.CONSTRUCT_PATTERNS.items():
            for match in pattern.finditer(text):
                protected_regions.append((match.start(), match.end(), match.group(0)))

        # Find bullet lists
        for match in self.BULLET_LIST_PATTERN.finditer(text):
            protected_regions.append((match.start(), match.end(), match.group(0)))

        # Sort regions by start position
        protected_regions.sort(key=lambda x: x[0])

        # Merge overlapping regions
        merged_regions: list[tuple[int, int, str]] = []
        for start, end, matched in protected_regions:
            if merged_regions and start <= merged_regions[-1][1]:
                # Overlapping - extend the previous region
                prev_start, prev_end, prev_text = merged_regions[-1]
                new_end = max(prev_end, end)
                merged_regions[-1] = (prev_start, new_end, text[prev_start:new_end])
            else:
                merged_regions.append((start, end, matched))

        # Build segments
        pos = 0
        for start, end, matched in merged_regions:
            # Add text before this construct
            if start > pos:
                before_text = text[pos:start].strip()
                if before_text:
                    segments.extend(self._split_paragraphs(before_text))

            # Add the protected construct as a single segment
            segments.append(matched.strip())
            pos = end

        # Add remaining text
        if pos < len(text):
            remaining = text[pos:].strip()
            if remaining:
                segments.extend(self._split_paragraphs(remaining))

        return segments

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split.

        Returns:
            List of paragraph strings.
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() + "\n\n" for p in paragraphs if p.strip()]

    def _chunk_paragraphs(self, text: str) -> list[ThreeGPPChunk]:
        """Chunk text using paragraph-based splitting when no clauses found.

        Args:
            text: Text to chunk.

        Returns:
            List of chunks.
        """
        chunks: list[ThreeGPPChunk] = []
        segments = self._split_preserving_constructs(text)

        current_chunk = ""
        for segment in segments:
            if len(current_chunk) + len(segment) > self._max_chunk_size:
                if current_chunk.strip():
                    chunks.append(
                        ThreeGPPChunk(
                            text=current_chunk.strip(),
                            chunk_type=ChunkType.TEXT,
                        )
                    )
                current_chunk = segment
            else:
                current_chunk += segment

        if current_chunk.strip():
            chunks.append(
                ThreeGPPChunk(
                    text=current_chunk.strip(),
                    chunk_type=ChunkType.TEXT,
                )
            )

        return chunks

    def _chunk_table(self, table: ParsedTable) -> list[ThreeGPPChunk]:
        """Chunk a table, keeping header with each chunk.

        Args:
            table: The parsed table.

        Returns:
            List of table chunks.
        """
        chunks: list[ThreeGPPChunk] = []

        # Format header row
        header_row = "| " + " | ".join(table.headers) + " |"
        separator = "| " + " | ".join(["---"] * len(table.headers)) + " |"
        header_text = f"{table.caption}\n{header_row}\n{separator}\n"

        # If table is small enough, return as single chunk
        if len(table.rows) <= self._table_row_group_size:
            rows_text = "\n".join("| " + " | ".join(row) + " |" for row in table.rows)
            chunks.append(
                ThreeGPPChunk(
                    text=header_text + rows_text,
                    chunk_type=ChunkType.TABLE,
                    table_caption=table.caption,
                )
            )
            return chunks

        # Split into row groups, repeating header
        for i in range(0, len(table.rows), self._table_row_group_size):
            row_group = table.rows[i : i + self._table_row_group_size]
            rows_text = "\n".join("| " + " | ".join(row) + " |" for row in row_group)

            part_label = f" (rows {i + 1}-{i + len(row_group)})"
            chunks.append(
                ThreeGPPChunk(
                    text=f"{table.caption}{part_label}\n{header_row}\n{separator}\n{rows_text}",
                    chunk_type=ChunkType.TABLE,
                    table_caption=table.caption,
                    metadata={"row_start": i, "row_end": i + len(row_group)},
                )
            )

        return chunks

    def process_file(self, file_path: Path | str, collection: str = "documents") -> list[Document]:
        """Process a file and return Document objects.

        Args:
            file_path: Path to the file.
            collection: Collection name for metadata.

        Returns:
            List of Document objects.
        """
        import hashlib

        path = Path(file_path)

        # Read file content
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        text = None
        for encoding in encodings:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError(f"Could not decode file: {path}")

        # Chunk the text
        chunks = self.chunk_text(text, _source=str(path))

        # Convert to Document objects
        documents: list[Document] = []
        for i, chunk in enumerate(chunks):
            # Generate document ID
            content = f"{path.absolute()}:{i}"
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

            metadata = {
                "source": str(path),
                "filename": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": chunk.chunk_type.value,
                "collection": collection,
            }

            if chunk.clause_number:
                metadata["clause_number"] = chunk.clause_number
            if chunk.clause_title:
                metadata["clause_title"] = chunk.clause_title
            if chunk.table_caption:
                metadata["table_caption"] = chunk.table_caption

            metadata.update(chunk.metadata)

            documents.append(
                Document(
                    text=chunk.text,
                    metadata=metadata,
                    document_id=doc_id,
                )
            )

        logger.info(f"Processed 3GPP document {path.name}: {len(documents)} chunks")
        return documents


def create_3gpp_chunker(
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    chunk_overlap: int = 100,
    preserve_constructs: bool = True,
    table_row_group_size: int = 10,
) -> ThreeGPPChunker:
    """Factory function to create a 3GPP chunker.

    Args:
        max_chunk_size: Maximum chunk size in characters.
        min_chunk_size: Minimum chunk size.
        chunk_overlap: Overlap between chunks.
        preserve_constructs: Keep procedural constructs together.
        table_row_group_size: Max rows per table chunk.

    Returns:
        Configured ThreeGPPChunker instance.
    """
    return ThreeGPPChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_constructs=preserve_constructs,
        table_row_group_size=table_row_group_size,
    )

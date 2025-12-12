"""Entity and relationship extraction for building knowledge graphs."""

import json
import logging
import re
from typing import Literal

from rag_service.infrastructure.neo4j_store import Entity, Relationship

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entities and relationships from text.

    Supports two modes:
    - rule_based: Uses regex patterns for known entity types
    - llm: Uses a local LLM (via Ollama) for extraction
    """

    # Common entity patterns for protocol/state machine documents
    STATE_PATTERNS = [
        r"\b(IDLE|ARMED|FLYING|LANDED|TAKEOFF|LANDING|RTL|LOITER|AUTO|MANUAL|STABILIZE|GUIDED)\b",
        r"\bstate[:\s]+([A-Z_]+)\b",
        r"\b([A-Z_]+)\s+state\b",
        r"transitions?\s+(?:to|from)\s+([A-Z_]+)",
    ]

    MESSAGE_PATTERNS = [
        r"\b(HEARTBEAT|COMMAND_LONG|COMMAND_INT|COMMAND_ACK|SYS_STATUS|GPS_RAW_INT|ATTITUDE)\b",
        r"\b(MISSION_\w+|SET_MODE|SET_POSITION_\w+|MANUAL_CONTROL)\b",
        r"\bmessage[:\s]+([A-Z_]+)\b",
        r"\b([A-Z_]+)\s+message\b",
    ]

    COMMAND_PATTERNS = [
        r"\b(ARM|DISARM|TAKEOFF|LAND|RTL|LOITER|GOTO|MISSION_START)\b",
        r"\bcommand[:\s]+([A-Z_]+)\b",
        r"\b(MAV_CMD_\w+)\b",
    ]

    PROTOCOL_PATTERNS = [
        r"\b(MAVLink|MAVLink2?|MAVROS)\b",
        r"\bprotocol[:\s]+(\w+)\b",
    ]

    # Relationship extraction patterns
    TRANSITION_PATTERNS = [
        r"(\w+)\s+(?:transitions?|goes?|moves?|changes?)\s+to\s+(\w+)",
        r"from\s+(\w+)\s+to\s+(\w+)",
        r"(\w+)\s*->\s*(\w+)",
        r"(\w+)\s*→\s*(\w+)",
    ]

    TRIGGER_PATTERNS = [
        r"(\w+)\s+(?:triggers?|causes?|initiates?)\s+(\w+)",
        r"(\w+)\s+(?:is triggered|is caused)\s+by\s+(\w+)",
        r"when\s+(\w+).*?(?:then|,)\s*(\w+)",
    ]

    REQUIRES_PATTERNS = [
        r"(\w+)\s+(?:requires?|needs?|depends?\s+on)\s+(\w+)",
        r"(\w+)\s+is\s+required\s+(?:for|by)\s+(\w+)",
    ]

    def __init__(
        self,
        mode: Literal["rule_based", "llm"] = "rule_based",
        llm_model: str = "llama3.2",
        domain_entities: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the entity extractor.

        Args:
            mode: Extraction mode - "rule_based" or "llm".
            llm_model: Ollama model name for LLM extraction.
            domain_entities: Optional dict of entity_type -> known entities.
        """
        self._mode = mode
        self._llm_model = llm_model
        self._domain_entities = domain_entities or {}

        # Compile patterns
        self._state_patterns = [re.compile(p, re.IGNORECASE) for p in self.STATE_PATTERNS]
        self._message_patterns = [re.compile(p, re.IGNORECASE) for p in self.MESSAGE_PATTERNS]
        self._command_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_PATTERNS]
        self._protocol_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROTOCOL_PATTERNS]
        self._transition_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TRANSITION_PATTERNS
        ]
        self._trigger_patterns = [re.compile(p, re.IGNORECASE) for p in self.TRIGGER_PATTERNS]
        self._requires_patterns = [re.compile(p, re.IGNORECASE) for p in self.REQUIRES_PATTERNS]

    def extract(
        self,
        text: str,
        source: str | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from.
            source: Optional source identifier for provenance.

        Returns:
            Tuple of (entities, relationships).
        """
        if self._mode == "llm":
            return self._extract_with_llm(text, source)
        return self._extract_with_rules(text, source)

    def _extract_with_rules(
        self,
        text: str,
        source: str | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract using regex pattern matching.

        Args:
            text: Text to extract from.
            source: Optional source identifier.

        Returns:
            Tuple of (entities, relationships).
        """
        entities: dict[str, Entity] = {}  # name -> Entity
        relationships: list[Relationship] = []

        # Extract entities by type
        self._extract_entity_type(
            text, self._state_patterns, "State", entities, source
        )
        self._extract_entity_type(
            text, self._message_patterns, "Message", entities, source
        )
        self._extract_entity_type(
            text, self._command_patterns, "Command", entities, source
        )
        self._extract_entity_type(
            text, self._protocol_patterns, "Protocol", entities, source
        )

        # Add domain-specific entities if provided
        for entity_type, entity_names in self._domain_entities.items():
            for name in entity_names:
                if name.lower() in text.lower() and name not in entities:
                    entities[name] = Entity(
                        name=name,
                        entity_type=entity_type,
                        source=source,
                    )

        # Extract relationships
        entity_names = set(entities.keys())

        # Transitions
        for pattern in self._transition_patterns:
            for match in pattern.finditer(text):
                src, tgt = match.group(1), match.group(2)
                if self._is_valid_entity(src, entity_names) and self._is_valid_entity(
                    tgt, entity_names
                ):
                    relationships.append(
                        Relationship(
                            source=src.upper(),
                            target=tgt.upper(),
                            relationship_type="TRANSITIONS_TO",
                        )
                    )

        # Triggers
        for pattern in self._trigger_patterns:
            for match in pattern.finditer(text):
                src, tgt = match.group(1), match.group(2)
                if self._is_valid_entity(src, entity_names) and self._is_valid_entity(
                    tgt, entity_names
                ):
                    relationships.append(
                        Relationship(
                            source=src.upper(),
                            target=tgt.upper(),
                            relationship_type="TRIGGERS",
                        )
                    )

        # Requirements
        for pattern in self._requires_patterns:
            for match in pattern.finditer(text):
                src, tgt = match.group(1), match.group(2)
                if self._is_valid_entity(src, entity_names) and self._is_valid_entity(
                    tgt, entity_names
                ):
                    relationships.append(
                        Relationship(
                            source=src.upper(),
                            target=tgt.upper(),
                            relationship_type="REQUIRES",
                        )
                    )

        # Deduplicate relationships
        seen = set()
        unique_relationships = []
        for rel in relationships:
            key = (rel.source, rel.target, rel.relationship_type)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        return list(entities.values()), unique_relationships

    def _extract_entity_type(
        self,
        text: str,
        patterns: list[re.Pattern],
        entity_type: str,
        entities: dict[str, Entity],
        source: str | None,
    ) -> None:
        """Extract entities of a specific type using patterns."""
        for pattern in patterns:
            for match in pattern.finditer(text):
                # Get the first capturing group or the whole match
                name = match.group(1) if match.lastindex else match.group(0)
                name = name.strip().upper()

                if len(name) >= 2 and name not in entities:
                    entities[name] = Entity(
                        name=name,
                        entity_type=entity_type,
                        source=source,
                    )

    def _is_valid_entity(self, name: str, known_entities: set[str]) -> bool:
        """Check if a name could be a valid entity."""
        name_upper = name.upper()
        # Check if it matches a known entity (case-insensitive)
        if any(name_upper == e.upper() for e in known_entities):
            return True
        # Check if it looks like an entity name (all caps, underscores)
        if re.match(r"^[A-Z][A-Z0-9_]*$", name_upper) and len(name_upper) >= 2:
            return True
        return False

    def _extract_with_llm(
        self,
        text: str,
        source: str | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract using an LLM via Ollama.

        Args:
            text: Text to extract from.
            source: Optional source identifier.

        Returns:
            Tuple of (entities, relationships).
        """
        try:
            import httpx

            prompt = f"""Extract entities and relationships from the following text about a protocol or state machine.

Text: "{text[:2000]}"

Return a JSON object with this exact structure:
{{
    "entities": [
        {{"name": "ENTITY_NAME", "type": "State|Message|Command|Protocol|Component"}}
    ],
    "relationships": [
        {{"source": "ENTITY1", "target": "ENTITY2", "type": "TRANSITIONS_TO|TRIGGERS|REQUIRES|SENDS|RECEIVES"}}
    ]
}}

Only include clearly defined entities and relationships. Use UPPERCASE for entity names."""

            response = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self._llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result_text = response.json().get("response", "{}")
                try:
                    result = json.loads(result_text)

                    entities = [
                        Entity(
                            name=e["name"],
                            entity_type=e.get("type", "Unknown"),
                            source=source,
                        )
                        for e in result.get("entities", [])
                    ]

                    relationships = [
                        Relationship(
                            source=r["source"],
                            target=r["target"],
                            relationship_type=r.get("type", "RELATED_TO"),
                        )
                        for r in result.get("relationships", [])
                    ]

                    logger.debug(
                        f"LLM extracted {len(entities)} entities, {len(relationships)} relationships"
                    )
                    return entities, relationships

                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON")

            logger.warning("LLM extraction failed, falling back to rule-based")
            return self._extract_with_rules(text, source)

        except Exception as e:
            logger.warning(f"LLM extraction error: {e}, falling back to rule-based")
            return self._extract_with_rules(text, source)

    def extract_from_documents(
        self,
        documents: list[dict[str, str]],
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract from multiple documents.

        Args:
            documents: List of dicts with "text" and optional "source" keys.

        Returns:
            Tuple of (all_entities, all_relationships) with deduplication.
        """
        all_entities: dict[str, Entity] = {}
        all_relationships: list[Relationship] = []
        seen_rels: set[tuple[str, str, str]] = set()

        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source")

            entities, relationships = self.extract(text, source)

            for entity in entities:
                if entity.name not in all_entities:
                    all_entities[entity.name] = entity

            for rel in relationships:
                key = (rel.source, rel.target, rel.relationship_type)
                if key not in seen_rels:
                    seen_rels.add(key)
                    all_relationships.append(rel)

        return list(all_entities.values()), all_relationships


class MAVLinkEntityExtractor(EntityExtractor):
    """Specialized extractor for MAVLink protocol documents.

    Pre-configured with MAVLink-specific entity types and patterns.
    """

    MAVLINK_STATES = [
        "IDLE",
        "ARMED",
        "FLYING",
        "LANDED",
        "TAKEOFF",
        "LANDING",
        "RTL",
        "LOITER",
        "AUTO",
        "MANUAL",
        "STABILIZE",
        "GUIDED",
        "PREFLIGHT",
        "BOOT",
        "CALIBRATING",
        "STANDBY",
        "ACTIVE",
        "CRITICAL",
        "EMERGENCY",
        "POWEROFF",
    ]

    MAVLINK_MESSAGES = [
        "HEARTBEAT",
        "SYS_STATUS",
        "SYSTEM_TIME",
        "PING",
        "COMMAND_LONG",
        "COMMAND_INT",
        "COMMAND_ACK",
        "MISSION_COUNT",
        "MISSION_ITEM",
        "MISSION_ACK",
        "GPS_RAW_INT",
        "ATTITUDE",
        "GLOBAL_POSITION_INT",
        "SET_MODE",
        "MANUAL_CONTROL",
        "RC_CHANNELS",
        "PARAM_VALUE",
        "PARAM_SET",
    ]

    MAVLINK_COMMANDS = [
        "MAV_CMD_NAV_TAKEOFF",
        "MAV_CMD_NAV_LAND",
        "MAV_CMD_NAV_RETURN_TO_LAUNCH",
        "MAV_CMD_NAV_WAYPOINT",
        "MAV_CMD_COMPONENT_ARM_DISARM",
        "MAV_CMD_DO_SET_MODE",
        "MAV_CMD_DO_CHANGE_SPEED",
        "MAV_CMD_PREFLIGHT_CALIBRATION",
    ]

    def __init__(
        self,
        mode: Literal["rule_based", "llm"] = "rule_based",
        llm_model: str = "llama3.2",
    ) -> None:
        """Initialize MAVLink-specific extractor."""
        domain_entities = {
            "State": self.MAVLINK_STATES,
            "Message": self.MAVLINK_MESSAGES,
            "Command": self.MAVLINK_COMMANDS,
        }

        super().__init__(
            mode=mode,
            llm_model=llm_model,
            domain_entities=domain_entities,
        )


def create_extractor(
    mode: Literal["rule_based", "llm"] = "rule_based",
    llm_model: str = "llama3.2",
    domain: str = "general",
) -> EntityExtractor:
    """Factory function to create an entity extractor.

    Args:
        mode: Extraction mode.
        llm_model: Ollama model for LLM mode.
        domain: Domain specialization ("general" or "mavlink").

    Returns:
        Configured EntityExtractor instance.
    """
    if domain == "mavlink":
        return MAVLinkEntityExtractor(mode=mode, llm_model=llm_model)
    return EntityExtractor(mode=mode, llm_model=llm_model)


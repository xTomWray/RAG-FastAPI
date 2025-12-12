"""Unit tests for the graph store implementations."""

import pytest

from rag_service.infrastructure.neo4j_store import (
    Entity,
    InMemoryGraphStore,
    Relationship,
    create_graph_store,
)


class TestInMemoryGraphStore:
    """Tests for InMemoryGraphStore class."""

    @pytest.fixture
    def store(self) -> InMemoryGraphStore:
        """Create an in-memory store for testing."""
        return InMemoryGraphStore()

    @pytest.fixture
    def sample_entities(self) -> list[Entity]:
        """Create sample entities for testing."""
        return [
            Entity(name="IDLE", entity_type="State", source="test.md"),
            Entity(name="ARMED", entity_type="State", source="test.md"),
            Entity(name="FLYING", entity_type="State", source="test.md"),
            Entity(name="HEARTBEAT", entity_type="Message", source="test.md"),
        ]

    @pytest.fixture
    def sample_relationships(self) -> list[Relationship]:
        """Create sample relationships for testing."""
        return [
            Relationship(source="IDLE", target="ARMED", relationship_type="TRANSITIONS_TO"),
            Relationship(source="ARMED", target="FLYING", relationship_type="TRANSITIONS_TO"),
            Relationship(source="FLYING", target="HEARTBEAT", relationship_type="REQUIRES"),
        ]

    def test_is_available(self, store: InMemoryGraphStore) -> None:
        """Test that in-memory store is always available."""
        assert store.is_available() is True

    def test_add_entity(self, store: InMemoryGraphStore) -> None:
        """Test adding a single entity."""
        entity = Entity(name="TEST", entity_type="State")
        result = store.add_entity(entity, collection="test")

        assert result == "TEST"

    def test_add_entities(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
    ) -> None:
        """Test adding multiple entities."""
        names = store.add_entities(sample_entities, collection="test")

        assert len(names) == 4
        assert "IDLE" in names
        assert "ARMED" in names

    def test_add_relationship(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
    ) -> None:
        """Test adding a relationship."""
        store.add_entities(sample_entities, collection="test")

        relationship = Relationship(
            source="IDLE",
            target="ARMED",
            relationship_type="TRANSITIONS_TO",
        )
        store.add_relationship(relationship, collection="test")

        # Verify by querying
        result = store.query_neighbors("IDLE", hops=1, collection="test")
        assert len(result.entities) > 0

    def test_add_relationships(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ) -> None:
        """Test adding multiple relationships."""
        store.add_entities(sample_entities, collection="test")
        store.add_relationships(sample_relationships, collection="test")

        result = store.query_neighbors("IDLE", hops=2, collection="test")
        entity_names = {e.name for e in result.entities}

        assert "ARMED" in entity_names

    def test_query_neighbors(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ) -> None:
        """Test querying neighbors."""
        store.add_entities(sample_entities, collection="test")
        store.add_relationships(sample_relationships, collection="test")

        # 1 hop from IDLE
        result = store.query_neighbors("IDLE", hops=1, collection="test")
        names = {e.name for e in result.entities}
        assert "ARMED" in names

        # 2 hops from IDLE
        result = store.query_neighbors("IDLE", hops=2, collection="test")
        names = {e.name for e in result.entities}
        assert "FLYING" in names

    def test_query_path(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ) -> None:
        """Test finding paths between entities."""
        store.add_entities(sample_entities, collection="test")
        store.add_relationships(sample_relationships, collection="test")

        result = store.query_path("IDLE", "FLYING", collection="test")

        assert len(result.paths) > 0
        assert result.paths[0][0] == "IDLE"
        assert result.paths[0][-1] == "FLYING"

    def test_query_by_type(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
    ) -> None:
        """Test querying entities by type."""
        store.add_entities(sample_entities, collection="test")

        states = store.query_by_type("State", collection="test")
        messages = store.query_by_type("Message", collection="test")

        assert len(states) == 3
        assert len(messages) == 1
        assert all(e.entity_type == "State" for e in states)

    def test_delete_collection(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
    ) -> None:
        """Test deleting a collection."""
        store.add_entities(sample_entities, collection="to_delete")

        count = store.delete_collection("to_delete")
        assert count == 4

        # Collection should be empty
        assert "to_delete" not in store.list_collections()

    def test_get_collection_stats(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
        sample_relationships: list[Relationship],
    ) -> None:
        """Test getting collection statistics."""
        store.add_entities(sample_entities, collection="stats_test")
        store.add_relationships(sample_relationships, collection="stats_test")

        stats = store.get_collection_stats("stats_test")

        assert stats["total_nodes"] == 4
        assert stats["total_relationships"] == 3
        assert "State" in stats["nodes"]
        assert "TRANSITIONS_TO" in stats["relationships"]

    def test_list_collections(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
    ) -> None:
        """Test listing collections."""
        store.add_entities(sample_entities[:2], collection="col1")
        store.add_entities(sample_entities[2:], collection="col2")

        collections = store.list_collections()

        assert "col1" in collections
        assert "col2" in collections

    def test_query_nonexistent_entity(
        self,
        store: InMemoryGraphStore,
    ) -> None:
        """Test querying for an entity that doesn't exist."""
        result = store.query_neighbors("NONEXISTENT", collection="test")

        assert len(result.entities) == 0
        assert len(result.paths) == 0

    def test_query_path_no_path(
        self,
        store: InMemoryGraphStore,
        sample_entities: list[Entity],
    ) -> None:
        """Test finding path when none exists."""
        # Add entities but no relationships
        store.add_entities(sample_entities, collection="test")

        result = store.query_path("IDLE", "FLYING", collection="test")

        assert len(result.paths) == 0

    def test_create_graph_store_memory(self) -> None:
        """Test factory function for in-memory store."""
        store = create_graph_store(backend="memory")
        assert isinstance(store, InMemoryGraphStore)

    def test_entity_properties(self, store: InMemoryGraphStore) -> None:
        """Test that entity properties are preserved."""
        entity = Entity(
            name="TEST",
            entity_type="State",
            properties={"description": "Test state", "priority": 1},
            source="test.md",
        )
        store.add_entity(entity, collection="test")

        entities = store.query_by_type("State", collection="test")
        assert len(entities) == 1
        assert entities[0].name == "TEST"


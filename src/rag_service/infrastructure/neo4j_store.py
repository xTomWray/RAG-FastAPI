"""Neo4j-based graph store implementation for knowledge graphs."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """An entity in the knowledge graph."""

    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


@dataclass
class Relationship:
    """A relationship between entities in the knowledge graph."""

    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSearchResult:
    """Result from a graph search query."""

    entities: list[Entity]
    relationships: list[Relationship]
    paths: list[list[str]]
    raw_result: list[dict[str, Any]]


class Neo4jGraphStore:
    """Neo4j adapter for knowledge graph operations.

    Provides CRUD operations for entities and relationships,
    as well as graph traversal queries.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        """Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI.
            user: Username for authentication.
            password: Password for authentication.
            database: Database name to use.
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None

    def _get_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self._uri, auth=(self._user, self._password)
                )
                # Verify connectivity
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self._uri}")
            except ImportError:
                raise ImportError(
                    "neo4j package is required for graph store. "
                    "Install with: pip install neo4j"
                )
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j: {e}")
                raise
        return self._driver

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def is_available(self) -> bool:
        """Check if Neo4j is available and connected."""
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            return True
        except Exception:
            return False

    def add_entity(self, entity: Entity, collection: str = "default") -> str:
        """Add a single entity to the graph.

        Args:
            entity: Entity to add.
            collection: Collection/namespace for the entity.

        Returns:
            The entity name (used as ID).
        """
        driver = self._get_driver()

        query = f"""
        MERGE (e:{entity.entity_type} {{name: $name, collection: $collection}})
        SET e += $properties
        SET e.source = $source
        RETURN e.name as name
        """

        with driver.session(database=self._database) as session:
            result = session.run(
                query,
                name=entity.name,
                collection=collection,
                properties=entity.properties,
                source=entity.source,
            )
            record = result.single()
            return record["name"] if record else entity.name

    def add_entities(self, entities: list[Entity], collection: str = "default") -> list[str]:
        """Add multiple entities to the graph.

        Args:
            entities: List of entities to add.
            collection: Collection/namespace for the entities.

        Returns:
            List of entity names.
        """
        if not entities:
            return []

        driver = self._get_driver()
        names = []

        with driver.session(database=self._database) as session:
            for entity in entities:
                query = f"""
                MERGE (e:{entity.entity_type} {{name: $name, collection: $collection}})
                SET e += $properties
                SET e.source = $source
                RETURN e.name as name
                """
                result = session.run(
                    query,
                    name=entity.name,
                    collection=collection,
                    properties=entity.properties,
                    source=entity.source,
                )
                record = result.single()
                names.append(record["name"] if record else entity.name)

        return names

    def add_relationship(self, relationship: Relationship, collection: str = "default") -> None:
        """Add a relationship between entities.

        Args:
            relationship: Relationship to add.
            collection: Collection/namespace.
        """
        driver = self._get_driver()

        # Dynamic relationship type
        query = f"""
        MATCH (source {{name: $source_name, collection: $collection}})
        MATCH (target {{name: $target_name, collection: $collection}})
        MERGE (source)-[r:{relationship.relationship_type}]->(target)
        SET r += $properties
        """

        with driver.session(database=self._database) as session:
            session.run(
                query,
                source_name=relationship.source,
                target_name=relationship.target,
                collection=collection,
                properties=relationship.properties,
            )

    def add_relationships(
        self, relationships: list[Relationship], collection: str = "default"
    ) -> None:
        """Add multiple relationships.

        Args:
            relationships: List of relationships to add.
            collection: Collection/namespace.
        """
        for rel in relationships:
            self.add_relationship(rel, collection)

    def query_neighbors(
        self,
        entity_name: str,
        hops: int = 2,
        collection: str = "default",
        direction: str = "both",
    ) -> GraphSearchResult:
        """Query neighboring entities within N hops.

        Args:
            entity_name: Starting entity name.
            hops: Maximum number of hops to traverse.
            collection: Collection to search in.
            direction: "outgoing", "incoming", or "both".

        Returns:
            GraphSearchResult with found entities and relationships.
        """
        driver = self._get_driver()

        # Build direction-aware pattern
        if direction == "outgoing":
            pattern = f"-[r*1..{hops}]->"
        elif direction == "incoming":
            pattern = f"<-[r*1..{hops}]-"
        else:
            pattern = f"-[r*1..{hops}]-"

        query = f"""
        MATCH (start {{name: $name, collection: $collection}})
        MATCH path = (start){pattern}(end)
        WHERE end.collection = $collection
        RETURN DISTINCT 
            end.name as entity_name,
            labels(end)[0] as entity_type,
            properties(end) as properties,
            [rel in relationships(path) | type(rel)] as rel_types,
            [node in nodes(path) | node.name] as path_nodes
        LIMIT 100
        """

        entities = []
        relationships = []
        paths = []
        raw_results = []

        with driver.session(database=self._database) as session:
            result = session.run(query, name=entity_name, collection=collection)

            for record in result:
                raw_results.append(dict(record))

                # Extract entity
                props = dict(record["properties"])
                props.pop("name", None)
                props.pop("collection", None)

                entities.append(
                    Entity(
                        name=record["entity_name"],
                        entity_type=record["entity_type"],
                        properties=props,
                    )
                )

                # Extract path
                paths.append(record["path_nodes"])

        return GraphSearchResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            raw_result=raw_results,
        )

    def query_path(
        self,
        start_entity: str,
        end_entity: str,
        collection: str = "default",
        max_hops: int = 5,
    ) -> GraphSearchResult:
        """Find paths between two entities.

        Args:
            start_entity: Starting entity name.
            end_entity: Target entity name.
            collection: Collection to search in.
            max_hops: Maximum path length.

        Returns:
            GraphSearchResult with found paths.
        """
        driver = self._get_driver()

        query = f"""
        MATCH path = shortestPath(
            (start {{name: $start, collection: $collection}})-[*1..{max_hops}]-
            (end {{name: $end, collection: $collection}})
        )
        RETURN 
            [node in nodes(path) | node.name] as path_nodes,
            [node in nodes(path) | labels(node)[0]] as path_types,
            [rel in relationships(path) | type(rel)] as rel_types
        LIMIT 10
        """

        paths = []
        raw_results = []

        with driver.session(database=self._database) as session:
            result = session.run(
                query, start=start_entity, end=end_entity, collection=collection
            )

            for record in result:
                raw_results.append(dict(record))
                paths.append(record["path_nodes"])

        return GraphSearchResult(
            entities=[],
            relationships=[],
            paths=paths,
            raw_result=raw_results,
        )

    def query_by_type(
        self,
        entity_type: str,
        collection: str = "default",
        limit: int = 100,
    ) -> list[Entity]:
        """Query all entities of a specific type.

        Args:
            entity_type: Type of entities to find.
            collection: Collection to search in.
            limit: Maximum number of results.

        Returns:
            List of matching entities.
        """
        driver = self._get_driver()

        query = f"""
        MATCH (e:{entity_type} {{collection: $collection}})
        RETURN e.name as name, labels(e)[0] as type, properties(e) as props
        LIMIT $limit
        """

        entities = []

        with driver.session(database=self._database) as session:
            result = session.run(query, collection=collection, limit=limit)

            for record in result:
                props = dict(record["props"])
                props.pop("name", None)
                props.pop("collection", None)

                entities.append(
                    Entity(
                        name=record["name"],
                        entity_type=record["type"],
                        properties=props,
                    )
                )

        return entities

    def query_relationships_from(
        self,
        entity_name: str,
        relationship_type: str | None = None,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        """Query relationships originating from an entity.

        Args:
            entity_name: Source entity name.
            relationship_type: Optional filter by relationship type.
            collection: Collection to search in.

        Returns:
            List of relationship dictionaries.
        """
        driver = self._get_driver()

        if relationship_type:
            query = f"""
            MATCH (source {{name: $name, collection: $collection}})-[r:{relationship_type}]->(target)
            RETURN source.name as source, type(r) as rel_type, target.name as target, properties(r) as props
            """
        else:
            query = """
            MATCH (source {name: $name, collection: $collection})-[r]->(target)
            RETURN source.name as source, type(r) as rel_type, target.name as target, properties(r) as props
            """

        results = []

        with driver.session(database=self._database) as session:
            result = session.run(query, name=entity_name, collection=collection)

            for record in result:
                results.append(
                    {
                        "source": record["source"],
                        "relationship_type": record["rel_type"],
                        "target": record["target"],
                        "properties": dict(record["props"]),
                    }
                )

        return results

    def delete_collection(self, collection: str) -> int:
        """Delete all entities and relationships in a collection.

        Args:
            collection: Collection to delete.

        Returns:
            Number of deleted nodes.
        """
        driver = self._get_driver()

        query = """
        MATCH (n {collection: $collection})
        DETACH DELETE n
        RETURN count(n) as deleted
        """

        with driver.session(database=self._database) as session:
            result = session.run(query, collection=collection)
            record = result.single()
            return record["deleted"] if record else 0

    def get_collection_stats(self, collection: str = "default") -> dict[str, Any]:
        """Get statistics about a collection.

        Args:
            collection: Collection to get stats for.

        Returns:
            Dictionary with node and relationship counts by type.
        """
        driver = self._get_driver()

        node_query = """
        MATCH (n {collection: $collection})
        RETURN labels(n)[0] as type, count(n) as count
        """

        rel_query = """
        MATCH (n {collection: $collection})-[r]->()
        RETURN type(r) as type, count(r) as count
        """

        stats = {"nodes": {}, "relationships": {}, "total_nodes": 0, "total_relationships": 0}

        with driver.session(database=self._database) as session:
            # Count nodes by type
            result = session.run(node_query, collection=collection)
            for record in result:
                stats["nodes"][record["type"]] = record["count"]
                stats["total_nodes"] += record["count"]

            # Count relationships by type
            result = session.run(rel_query, collection=collection)
            for record in result:
                stats["relationships"][record["type"]] = record["count"]
                stats["total_relationships"] += record["count"]

        return stats

    def list_collections(self) -> list[str]:
        """List all collections in the database.

        Returns:
            List of collection names.
        """
        driver = self._get_driver()

        query = """
        MATCH (n)
        WHERE n.collection IS NOT NULL
        RETURN DISTINCT n.collection as collection
        """

        collections = []

        with driver.session(database=self._database) as session:
            result = session.run(query)
            for record in result:
                collections.append(record["collection"])

        return collections


class InMemoryGraphStore:
    """In-memory graph store using NetworkX for environments without Neo4j.

    Provides the same interface as Neo4jGraphStore but stores data in memory.
    Useful for testing and lightweight deployments.
    """

    def __init__(self) -> None:
        """Initialize in-memory graph store."""
        try:
            import networkx as nx

            self._graphs: dict[str, nx.DiGraph] = {}
            self._nx = nx
        except ImportError:
            raise ImportError(
                "networkx package is required for in-memory graph store. "
                "Install with: pip install networkx"
            )

    def _get_graph(self, collection: str) -> "nx.DiGraph":
        """Get or create a graph for a collection."""
        if collection not in self._graphs:
            self._graphs[collection] = self._nx.DiGraph()
        return self._graphs[collection]

    def is_available(self) -> bool:
        """Always available for in-memory store."""
        return True

    def close(self) -> None:
        """No-op for in-memory store."""
        pass

    def add_entity(self, entity: Entity, collection: str = "default") -> str:
        """Add a single entity to the graph."""
        graph = self._get_graph(collection)
        graph.add_node(
            entity.name,
            entity_type=entity.entity_type,
            source=entity.source,
            **entity.properties,
        )
        return entity.name

    def add_entities(self, entities: list[Entity], collection: str = "default") -> list[str]:
        """Add multiple entities to the graph."""
        return [self.add_entity(e, collection) for e in entities]

    def add_relationship(self, relationship: Relationship, collection: str = "default") -> None:
        """Add a relationship between entities."""
        graph = self._get_graph(collection)
        graph.add_edge(
            relationship.source,
            relationship.target,
            relationship_type=relationship.relationship_type,
            **relationship.properties,
        )

    def add_relationships(
        self, relationships: list[Relationship], collection: str = "default"
    ) -> None:
        """Add multiple relationships."""
        for rel in relationships:
            self.add_relationship(rel, collection)

    def query_neighbors(
        self,
        entity_name: str,
        hops: int = 2,
        collection: str = "default",
        direction: str = "both",
    ) -> GraphSearchResult:
        """Query neighboring entities within N hops."""
        graph = self._get_graph(collection)

        if entity_name not in graph:
            return GraphSearchResult([], [], [], [])

        # BFS to find neighbors within hops
        visited = {entity_name}
        current_level = {entity_name}
        entities = []
        paths = []

        for hop in range(hops):
            next_level = set()
            for node in current_level:
                # Get neighbors based on direction
                if direction == "outgoing":
                    neighbors = set(graph.successors(node))
                elif direction == "incoming":
                    neighbors = set(graph.predecessors(node))
                else:
                    neighbors = set(graph.successors(node)) | set(graph.predecessors(node))

                for neighbor in neighbors - visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)

                    node_data = graph.nodes[neighbor]
                    entities.append(
                        Entity(
                            name=neighbor,
                            entity_type=node_data.get("entity_type", "Unknown"),
                            properties={
                                k: v
                                for k, v in node_data.items()
                                if k not in ("entity_type", "source")
                            },
                            source=node_data.get("source"),
                        )
                    )
                    paths.append([entity_name, neighbor])

            current_level = next_level

        return GraphSearchResult(
            entities=entities,
            relationships=[],
            paths=paths,
            raw_result=[],
        )

    def query_path(
        self,
        start_entity: str,
        end_entity: str,
        collection: str = "default",
        max_hops: int = 5,
    ) -> GraphSearchResult:
        """Find paths between two entities."""
        graph = self._get_graph(collection)

        if start_entity not in graph or end_entity not in graph:
            return GraphSearchResult([], [], [], [])

        try:
            # Find shortest path
            path = self._nx.shortest_path(
                graph, start_entity, end_entity, weight=None
            )
            return GraphSearchResult(
                entities=[],
                relationships=[],
                paths=[path],
                raw_result=[{"path": path}],
            )
        except self._nx.NetworkXNoPath:
            return GraphSearchResult([], [], [], [])

    def query_by_type(
        self,
        entity_type: str,
        collection: str = "default",
        limit: int = 100,
    ) -> list[Entity]:
        """Query all entities of a specific type."""
        graph = self._get_graph(collection)
        entities = []

        for node, data in graph.nodes(data=True):
            if data.get("entity_type") == entity_type:
                entities.append(
                    Entity(
                        name=node,
                        entity_type=entity_type,
                        properties={
                            k: v
                            for k, v in data.items()
                            if k not in ("entity_type", "source")
                        },
                        source=data.get("source"),
                    )
                )
                if len(entities) >= limit:
                    break

        return entities

    def delete_collection(self, collection: str) -> int:
        """Delete all entities in a collection."""
        if collection in self._graphs:
            count = self._graphs[collection].number_of_nodes()
            del self._graphs[collection]
            return count
        return 0

    def get_collection_stats(self, collection: str = "default") -> dict[str, Any]:
        """Get statistics about a collection."""
        graph = self._get_graph(collection)

        node_types: dict[str, int] = {}
        rel_types: dict[str, int] = {}

        for node, data in graph.nodes(data=True):
            t = data.get("entity_type", "Unknown")
            node_types[t] = node_types.get(t, 0) + 1

        for u, v, data in graph.edges(data=True):
            t = data.get("relationship_type", "Unknown")
            rel_types[t] = rel_types.get(t, 0) + 1

        return {
            "nodes": node_types,
            "relationships": rel_types,
            "total_nodes": graph.number_of_nodes(),
            "total_relationships": graph.number_of_edges(),
        }

    def list_collections(self) -> list[str]:
        """List all collections."""
        return list(self._graphs.keys())


def create_graph_store(
    backend: str = "neo4j",
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
) -> Neo4jGraphStore | InMemoryGraphStore:
    """Factory function to create a graph store.

    Args:
        backend: "neo4j" or "memory".
        uri: Neo4j URI (only for neo4j backend).
        user: Neo4j username (only for neo4j backend).
        password: Neo4j password (only for neo4j backend).

    Returns:
        Configured graph store instance.
    """
    if backend == "memory":
        return InMemoryGraphStore()

    store = Neo4jGraphStore(uri=uri, user=user, password=password)

    # Check if Neo4j is available, fall back to in-memory
    if not store.is_available():
        logger.warning("Neo4j not available, falling back to in-memory graph store")
        return InMemoryGraphStore()

    return store


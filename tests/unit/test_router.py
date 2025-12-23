"""Unit tests for the query router."""

import pytest

from rag_service.core.router import QueryRouter, create_router


class TestQueryRouter:
    """Tests for QueryRouter class."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        """Create a router for testing."""
        return create_router(mode="pattern", default_strategy="vector")

    def test_semantic_query_classification(self, router: QueryRouter) -> None:
        """Test that semantic queries are classified correctly."""
        semantic_queries = [
            "What is MAVLink?",
            "Explain the protocol structure",
            "Describe the authentication mechanism",
            "Define the heartbeat message",
            "Tell me about drone communication",
            "How does MAVLink work?",
        ]

        for query in semantic_queries:
            result = router.classify(query)
            assert result in ("vector", "hybrid"), f"'{query}' should be vector or hybrid, got {result}"

    def test_relational_query_classification(self, router: QueryRouter) -> None:
        """Test that relational queries are classified correctly."""
        relational_queries = [
            "What states follow ARMED?",
            "What transitions from IDLE to FLYING?",
            "What triggers the TAKEOFF command?",
            "Path from ARMED to LANDED",
            "What depends on HEARTBEAT?",
            "What connects to the ARM command?",
        ]

        for query in relational_queries:
            result = router.classify(query)
            assert result in ("graph", "hybrid"), f"'{query}' should be graph or hybrid, got {result}"

    def test_hybrid_query_classification(self, router: QueryRouter) -> None:
        """Test that complex queries are classified as hybrid."""
        hybrid_queries = [
            "Explain the relationship between ARM and TAKEOFF",
            "Describe the transition flow from IDLE to FLYING",
            "What is the security of state transitions?",
        ]

        for query in hybrid_queries:
            result = router.classify(query)
            assert result == "hybrid", f"'{query}' should be hybrid, got {result}"

    def test_default_strategy(self) -> None:
        """Test that ambiguous queries use default strategy."""
        router = create_router(default_strategy="vector")
        result = router.classify("random text without patterns")
        assert result == "vector"

        router = create_router(default_strategy="hybrid")
        result = router.classify("random text without patterns")
        assert result == "hybrid"

    def test_get_strategy_explanation(self, router: QueryRouter) -> None:
        """Test that explanation includes relevant information."""
        explanation = router.get_strategy_explanation("What states follow ARMED?")

        assert "strategy" in explanation
        assert "reasoning" in explanation
        assert "matched_relational_patterns" in explanation
        assert "matched_semantic_patterns" in explanation

    def test_case_insensitivity(self, router: QueryRouter) -> None:
        """Test that classification is case-insensitive."""
        query_lower = "what states follow armed?"
        query_upper = "WHAT STATES FOLLOW ARMED?"
        query_mixed = "What States Follow Armed?"

        result_lower = router.classify(query_lower)
        result_upper = router.classify(query_upper)
        result_mixed = router.classify(query_mixed)

        # All should produce the same classification
        assert result_lower == result_upper == result_mixed

    def test_create_router_factory(self) -> None:
        """Test the factory function."""
        router = create_router(
            mode="pattern",
            default_strategy="graph",
            llm_model="test-model",
        )

        assert router._mode == "pattern"
        assert router._default_strategy == "graph"
        assert router._llm_model == "test-model"

    def test_transition_patterns(self, router: QueryRouter) -> None:
        """Test various transition-related patterns."""
        queries = [
            ("IDLE transitions to ARMED", "graph"),
            ("goes from state A to state B", "graph"),
            ("moves to the next state", "graph"),
            ("changes to FLYING", "graph"),
        ]

        for query, expected_type in queries:
            result = router.classify(query)
            assert result in (expected_type, "hybrid"), f"'{query}' failed"

    def test_dependency_patterns(self, router: QueryRouter) -> None:
        """Test dependency-related patterns."""
        queries = [
            "What requires HEARTBEAT?",
            "FLYING depends on GPS",
            "ARM needs PREFLIGHT check",
        ]

        for query in queries:
            result = router.classify(query)
            assert result in ("graph", "hybrid"), f"'{query}' should involve graph"


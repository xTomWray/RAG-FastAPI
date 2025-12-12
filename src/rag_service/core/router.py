"""Smart query router for deciding between vector and graph search."""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

QueryStrategy = Literal["vector", "graph", "hybrid"]


class QueryRouter:
    """Classifies queries to determine optimal retrieval strategy.

    Uses pattern matching and optionally an LLM classifier to route
    queries to the most appropriate retrieval method:
    - vector: Semantic similarity search (default)
    - graph: Relationship traversal in knowledge graph
    - hybrid: Both vector and graph search combined
    """

    # Patterns that indicate relational/graph queries
    RELATIONAL_PATTERNS = [
        # Transition/flow patterns
        r"what.*(?:follow|precede|trigger|cause|lead|come after|come before)",
        r"(?:transition|path|route|flow).*(?:from|to|between)",
        r"(?:after|before|following|preceding).*(?:state|step|stage|phase)",
        # Relationship patterns
        r"(?:connect|relate|link|associate).*(?:to|with|between)",
        r"(?:depend|require|need).*(?:on|for|by)",
        r"(?:how|what).*(?:related|connected|linked)",
        # Graph traversal patterns
        r"(?:all|list|show).*(?:that|which).*(?:lead|go|connect)",
        r"(?:path|route|way).*(?:from|to|between)",
        r"(?:neighbor|adjacent|next|previous).*(?:state|node|entity)",
        # Specific entity relationship queries
        r"what.*(?:can|could|might|will).*(?:happen|occur|result)",
        r"(?:which|what).*(?:states?|commands?|messages?).*(?:trigger|cause|follow)",
    ]

    # Patterns that indicate semantic/vector queries
    SEMANTIC_PATTERNS = [
        r"^what is",
        r"^explain",
        r"^describe",
        r"^define",
        r"^tell me about",
        r"^how does.*work",
        r"^why",
        r"^overview of",
        r"^summary of",
        r"^meaning of",
    ]

    # Patterns that indicate hybrid queries (need both)
    HYBRID_PATTERNS = [
        r"explain.*(?:relationship|connection|transition)",
        r"describe.*(?:flow|path|sequence)",
        r"(?:detail|elaborate).*(?:how|why).*(?:relate|connect|transition)",
        r"(?:security|vulnerability|attack).*(?:path|transition|state)",
    ]

    def __init__(
        self,
        mode: Literal["pattern", "llm"] = "pattern",
        default_strategy: QueryStrategy = "vector",
        llm_model: str | None = None,
    ) -> None:
        """Initialize the query router.

        Args:
            mode: Classification mode - "pattern" for regex-based, "llm" for LLM-based.
            default_strategy: Default strategy when classification is uncertain.
            llm_model: Ollama model name for LLM-based classification.
        """
        self._mode = mode
        self._default_strategy = default_strategy
        self._llm_model = llm_model or "llama3.2"

        # Compile patterns for efficiency
        self._relational_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.RELATIONAL_PATTERNS
        ]
        self._semantic_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SEMANTIC_PATTERNS
        ]
        self._hybrid_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HYBRID_PATTERNS
        ]

    def classify(self, query: str) -> QueryStrategy:
        """Classify a query to determine the best retrieval strategy.

        Args:
            query: The user's query string.

        Returns:
            The recommended strategy: "vector", "graph", or "hybrid".
        """
        if self._mode == "llm":
            return self._classify_with_llm(query)
        return self._classify_with_patterns(query)

    def _classify_with_patterns(self, query: str) -> QueryStrategy:
        """Classify query using regex pattern matching.

        Args:
            query: The user's query string.

        Returns:
            The recommended strategy.
        """
        query_lower = query.lower().strip()

        # Check hybrid patterns first (most specific)
        for pattern in self._hybrid_patterns:
            if pattern.search(query_lower):
                logger.debug(f"Query matched hybrid pattern: {pattern.pattern}")
                return "hybrid"

        # Check relational patterns
        relational_score = sum(
            1 for p in self._relational_patterns if p.search(query_lower)
        )

        # Check semantic patterns
        semantic_score = sum(
            1 for p in self._semantic_patterns if p.search(query_lower)
        )

        logger.debug(
            f"Query scores - relational: {relational_score}, semantic: {semantic_score}"
        )

        # Decision logic
        if relational_score > 0 and semantic_score == 0:
            return "graph"
        elif relational_score > semantic_score:
            return "hybrid"
        elif relational_score > 0:
            return "hybrid"

        return self._default_strategy

    def _classify_with_llm(self, query: str) -> QueryStrategy:
        """Classify query using an LLM.

        Args:
            query: The user's query string.

        Returns:
            The recommended strategy.
        """
        try:
            import httpx

            prompt = f"""Classify the following query for a RAG system that has both:
1. Vector search (semantic similarity in documents)
2. Graph search (entity relationships, state transitions, paths)

Query: "{query}"

Respond with ONLY one word: "vector", "graph", or "hybrid"

- Use "vector" for general questions, explanations, definitions
- Use "graph" for relationship questions, transitions, paths, what-follows-what
- Use "hybrid" for complex questions needing both context and relationships"""

            response = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self._llm_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=10.0,
            )

            if response.status_code == 200:
                result = response.json().get("response", "").strip().lower()
                if result in ("vector", "graph", "hybrid"):
                    logger.debug(f"LLM classified query as: {result}")
                    return result  # type: ignore

            logger.warning("LLM classification failed, falling back to pattern matching")
            return self._classify_with_patterns(query)

        except Exception as e:
            logger.warning(f"LLM classification error: {e}, falling back to patterns")
            return self._classify_with_patterns(query)

    def get_strategy_explanation(self, query: str) -> dict[str, str | list[str]]:
        """Get detailed explanation of why a strategy was chosen.

        Args:
            query: The user's query string.

        Returns:
            Dictionary with strategy and matching patterns.
        """
        query_lower = query.lower().strip()

        matched_relational = [
            p.pattern for p in self._relational_patterns if p.search(query_lower)
        ]
        matched_semantic = [
            p.pattern for p in self._semantic_patterns if p.search(query_lower)
        ]
        matched_hybrid = [
            p.pattern for p in self._hybrid_patterns if p.search(query_lower)
        ]

        strategy = self.classify(query)

        return {
            "strategy": strategy,
            "matched_relational_patterns": matched_relational,
            "matched_semantic_patterns": matched_semantic,
            "matched_hybrid_patterns": matched_hybrid,
            "reasoning": self._get_reasoning(
                strategy, matched_relational, matched_semantic, matched_hybrid
            ),
        }

    def _get_reasoning(
        self,
        strategy: QueryStrategy,
        relational: list[str],
        semantic: list[str],
        hybrid: list[str],
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        if hybrid:
            return f"Query matches hybrid patterns, needs both semantic context and graph relationships"
        elif strategy == "graph":
            return f"Query is asking about relationships/transitions ({len(relational)} relational patterns matched)"
        elif strategy == "hybrid" and relational:
            return f"Query has both semantic and relational aspects ({len(semantic)} semantic, {len(relational)} relational)"
        else:
            return f"Query is a general/explanatory question best suited for semantic search"


def create_router(
    mode: Literal["pattern", "llm"] = "pattern",
    default_strategy: QueryStrategy = "vector",
    llm_model: str | None = None,
) -> QueryRouter:
    """Factory function to create a query router.

    Args:
        mode: Classification mode.
        default_strategy: Default strategy for ambiguous queries.
        llm_model: Ollama model for LLM mode.

    Returns:
        Configured QueryRouter instance.
    """
    return QueryRouter(mode=mode, default_strategy=default_strategy, llm_model=llm_model)


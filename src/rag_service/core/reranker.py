"""Cross-encoder reranker service for improving search result relevance.

Cross-encoders process query-document pairs together, providing more accurate
relevance scores than bi-encoder similarity. Used as a second-stage ranker
on top-k results from initial vector search.

Typical workflow:
1. Initial retrieval: Vector search returns top_k candidates (fast, ~100 results)
2. Reranking: Cross-encoder scores each candidate (slower, but more accurate)
3. Return: Top reranked results by cross-encoder score
"""

import logging
import time
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from rag_service.core.gpu_diagnostics import get_diagnostic_logger
from rag_service.core.stats import get_stats_collector

logger = logging.getLogger(__name__)
diag_log = get_diagnostic_logger()

# Precision type
PrecisionType = Literal["fp32", "fp16", "auto"]


class RerankerProvider(Protocol):
    """Protocol for reranker providers."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. None returns all.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        ...

    def score_pairs(
        self,
        query: str,
        documents: list[str],
    ) -> NDArray[np.float32]:
        """Score query-document pairs.

        Args:
            query: The search query.
            documents: List of document texts.

        Returns:
            Array of relevance scores for each document.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class CrossEncoderReranker:
    """Reranker service using cross-encoder models.

    Cross-encoders process query and document together through a transformer,
    producing a single relevance score. More accurate than bi-encoder similarity
    but slower (O(n) forward passes vs O(1) for bi-encoder).

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality (default)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality, slower
    - BAAI/bge-reranker-base: Good multilingual support
    - BAAI/bge-reranker-large: Best quality, slowest
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        # GPU safeguard settings
        enable_gpu_safeguards: bool = True,
        max_memory_percent: float = 80.0,
        max_temperature_c: float = 75.0,
        # Precision settings
        precision: PrecisionType = "auto",
    ) -> None:
        """Initialize the reranker service.

        Args:
            model_name: HuggingFace model ID for cross-encoder.
            device: Device to run inference on ("auto", "cpu", "cuda", "mps").
            batch_size: Batch size for scoring pairs.
            max_length: Maximum token length for input pairs.
            enable_gpu_safeguards: Enable GPU memory/temperature monitoring.
            max_memory_percent: Maximum GPU memory usage before throttling.
            max_temperature_c: Maximum GPU temperature before throttling.
            precision: Floating point precision ("fp32", "fp16", "auto").
        """
        from sentence_transformers import CrossEncoder

        resolved_device = self._resolve_device(device)
        self._is_gpu = resolved_device in ("cuda", "mps")

        # Resolve precision
        resolved_precision = self._resolve_precision(precision, resolved_device)
        self._precision = resolved_precision

        # Log model loading
        diag_log.start_operation(
            "reranker_load",
            model_name=model_name,
            device=resolved_device,
            precision=resolved_precision,
        )
        logger.info(
            f"Loading reranker model {model_name} on {resolved_device} ({resolved_precision})"
        )

        try:
            load_start = time.perf_counter()

            # CrossEncoder model kwargs
            model_kwargs: dict[str, Any] = {}
            if resolved_precision == "fp16" and self._is_gpu:
                import torch

                model_kwargs["torch_dtype"] = torch.float16

            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=resolved_device,
                automodel_args=model_kwargs,
            )

            load_duration = (time.perf_counter() - load_start) * 1000
            diag_log.end_operation("reranker_load", success=True)
            logger.info(f"Reranker model loaded in {load_duration:.0f}ms")

        except Exception as e:
            diag_log.end_operation("reranker_load", success=False, error=str(e))
            logger.error(f"Failed to load reranker model: {e}")
            raise

        self._batch_size = batch_size
        self._max_length = max_length
        self._model_name = model_name
        self._device = resolved_device

        # GPU safeguard settings
        self._enable_gpu_safeguards = enable_gpu_safeguards
        self._max_memory_percent = max_memory_percent
        self._max_temperature_c = max_temperature_c

        if self._is_gpu and enable_gpu_safeguards:
            logger.info(
                f"GPU safeguards enabled for reranker: max_memory={max_memory_percent}%, "
                f"max_temp={max_temperature_c}°C"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string to actual device."""
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _resolve_precision(precision: PrecisionType, device: str) -> str:
        """Resolve precision based on device capabilities."""
        if precision != "auto":
            return precision

        if device == "cpu":
            return "fp32"

        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                cc = torch.cuda.get_device_capability(0)
                if cc[0] > 5 or (cc[0] == 5 and cc[1] >= 3):
                    return "fp16"
            elif device == "mps":
                return "fp16"
        except Exception:
            pass

        return "fp32"

    def score_pairs(
        self,
        query: str,
        documents: list[str],
    ) -> NDArray[np.float32]:
        """Score query-document pairs.

        Args:
            query: The search query.
            documents: List of document texts.

        Returns:
            Array of relevance scores for each document.
        """
        if not documents:
            return np.array([], dtype=np.float32)

        start_time = time.perf_counter()
        success = True

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Score pairs
            if self._is_gpu and self._enable_gpu_safeguards:
                scores = self._score_pairs_safe(pairs)
            else:
                scores = self._model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=len(pairs) > 100,
                )

            return np.array(scores, dtype=np.float32)

        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            stats = get_stats_collector()
            stats.record_rerank(
                duration_ms=duration_ms,
                num_documents=len(documents),
                success=success,
            )

    def _score_pairs_safe(self, pairs: list[list[str]]) -> NDArray[np.float32]:
        """Score pairs with GPU memory safeguards.

        Args:
            pairs: List of [query, document] pairs.

        Returns:
            Array of scores.
        """
        from rag_service.core.gpu_utils import clear_gpu_memory, get_gpu_status

        logger.debug(f"Reranking {len(pairs)} pairs with GPU safeguards")

        # Clear memory before starting
        clear_gpu_memory()

        # Check GPU status
        status = get_gpu_status()
        if status.available and status.memory_percent > self._max_memory_percent:
            logger.warning(
                f"GPU memory high ({status.memory_percent:.1f}%), reducing batch size for reranking"
            )
            batch_size = max(1, self._batch_size // 2)
        else:
            batch_size = self._batch_size

        # Process in batches
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_scores = self._model.predict(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
            )
            all_scores.extend(batch_scores)

            # Check thermal limits
            if self._is_gpu:
                status = get_gpu_status()
                if status.temperature_c and status.temperature_c > self._max_temperature_c:
                    import time as time_module

                    logger.warning(f"GPU temp {status.temperature_c}°C exceeds limit, cooling...")
                    time_module.sleep(0.5)

        clear_gpu_memory()
        return np.array(all_scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. None returns all.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        if not documents:
            return []

        # Get scores
        scores = self.score_pairs(query, documents)

        # Create index-score pairs and sort by score descending
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def get_device_info(self) -> dict[str, str]:
        """Get information about the device being used.

        Returns:
            Dictionary with device information.
        """
        info = {
            "device": self._device,
            "model": self._model_name,
            "precision": self._precision,
        }

        try:
            import torch

            if self._device == "cuda" and torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = (
                    f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
                )
        except Exception:
            pass

        return info


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 512,
    enable_gpu_safeguards: bool = True,
    max_memory_percent: float = 80.0,
    max_temperature_c: float = 75.0,
    precision: PrecisionType = "auto",
) -> CrossEncoderReranker:
    """Factory function to create a reranker service.

    Args:
        model_name: HuggingFace model ID for cross-encoder.
        device: Device to run inference on.
        batch_size: Batch size for scoring pairs.
        max_length: Maximum token length for input pairs.
        enable_gpu_safeguards: Enable GPU memory/temperature monitoring.
        max_memory_percent: Maximum GPU memory usage before throttling.
        max_temperature_c: Maximum GPU temperature before throttling.
        precision: Floating point precision.

    Returns:
        Configured reranker service instance.
    """
    return CrossEncoderReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        enable_gpu_safeguards=enable_gpu_safeguards,
        max_memory_percent=max_memory_percent,
        max_temperature_c=max_temperature_c,
        precision=precision,
    )

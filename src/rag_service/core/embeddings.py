"""Embedding service for generating document and query embeddings.

Includes GPU memory safeguards to prevent system crashes during large
embedding operations. Features power management to prevent transient
power spikes that can crash systems with high-power GPUs (RTX 30/40 series).

Supports FP16 (half precision) for reduced memory and power usage.
"""

import logging
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from rag_service.core.gpu_diagnostics import get_diagnostic_logger
from rag_service.core.stats import get_stats_collector

logger = logging.getLogger(__name__)
diag_log = get_diagnostic_logger()

# Precision type
PrecisionType = Literal["fp32", "fp16", "auto"]


def _try_apply_power_limit(target_watts: int | None) -> bool:
    """Attempt to apply GPU power limit (requires admin).

    Args:
        target_watts: Target power limit in watts, or None to skip.

    Returns:
        True if power limit was set successfully.
    """
    if target_watts is None:
        return False

    try:
        from rag_service.core.gpu_power import get_power_info, set_power_limit

        info = get_power_info()
        if info and info["power_limit"] != target_watts:
            success, msg = set_power_limit(target_watts)
            if success:
                logger.info(f"Applied GPU power limit: {target_watts}W")
            else:
                logger.debug(f"Could not set power limit: {msg}")
            return success
    except Exception as e:
        logger.debug(f"Power limit not applied: {e}")

    return False


def _configure_torch_performance(
    enable_tf32: bool = True, enable_cudnn_benchmark: bool = True
) -> None:
    """Configure PyTorch for maximum GPU performance.

    Args:
        enable_tf32: Enable TF32 for matmul (Ampere+ GPUs). ~3x faster.
        enable_cudnn_benchmark: Enable cuDNN benchmark mode.

    Enables:
    - TF32 for matmul (19x Tensor Core throughput on Ampere)
    - cuDNN benchmark mode (finds fastest algorithms)
    - cuDNN TF32 (faster convolutions)
    """
    try:
        import torch

        if torch.cuda.is_available():
            opts_enabled = []

            if enable_tf32:
                # Enable TF32 for matrix multiplications (Ampere+)
                # This gives ~3x speedup with <0.1% precision loss
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                opts_enabled.append("TF32")

            if enable_cudnn_benchmark:
                # Enable cuDNN benchmark mode - finds fastest algorithms
                # Best for fixed input sizes (like batched embeddings)
                torch.backends.cudnn.benchmark = True
                opts_enabled.append("cuDNN_benchmark")

            # Disable debug features for speed
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.emit_nvtx(False)  # type: ignore[no-untyped-call]

            if opts_enabled:
                logger.info(f"PyTorch performance optimizations enabled: {', '.join(opts_enabled)}")
    except Exception as e:
        logger.debug(f"Could not configure PyTorch performance: {e}")


def _warmup_gpu(model: Any, batch_sizes: list[int] | None = None) -> None:
    """Warm up GPU to avoid cold-start power spikes.

    Args:
        model: SentenceTransformer model.
        batch_sizes: Progressive batch sizes for warmup.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4]

    warmup_text = "GPU warmup text for power state transition."

    try:
        import torch

        logger.info("Warming up GPU to reduce power spikes...")

        for batch_size in batch_sizes:
            texts = [warmup_text] * batch_size
            _ = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(0.05)

        logger.info("GPU warmup complete")
    except Exception as e:
        logger.debug(f"GPU warmup skipped: {e}")


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding array with shape (embedding_dim,).
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        ...


class SentenceTransformerEmbedding:
    """Embedding service using sentence-transformers library.

    Supports CUDA (Windows/Linux), MPS (macOS), and CPU fallback.
    Includes GPU memory safeguards to prevent system crashes.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
        # GPU safeguard settings
        enable_gpu_safeguards: bool = True,
        max_memory_percent: float = 80.0,
        max_temperature_c: float = 75.0,
        inter_batch_delay: float = 0.05,
        adaptive_batch_size: bool = True,
        min_batch_size: int = 4,
        # Power management settings
        gpu_power_limit_watts: int | None = None,
        enable_gpu_warmup: bool = True,
        # Precision settings
        precision: PrecisionType = "auto",
        # Performance settings
        enable_tf32: bool = True,
        enable_cudnn_benchmark: bool = True,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model ID for embeddings.
            device: Device to run inference on ("auto", "cpu", "cuda", "mps").
            batch_size: Batch size for embedding generation.
            enable_gpu_safeguards: Enable GPU memory/temperature monitoring.
            max_memory_percent: Maximum GPU memory usage before throttling.
            max_temperature_c: Maximum GPU temperature before throttling.
            inter_batch_delay: Seconds to pause between batches.
            adaptive_batch_size: Automatically reduce batch size under pressure.
            min_batch_size: Minimum batch size when adapting.
            gpu_power_limit_watts: Target GPU power limit (requires admin). None to skip.
            enable_gpu_warmup: Warm up GPU after loading to reduce power spikes.
            precision: Floating point precision ("fp32", "fp16", "auto").
                       FP16 uses ~50% less memory and ~30% less power.
            enable_tf32: Enable TF32 for matmul on Ampere+ GPUs (~3x faster).
            enable_cudnn_benchmark: Enable cuDNN benchmark mode.
        """
        from sentence_transformers import SentenceTransformer

        resolved_device = self._resolve_device(device)
        self._is_gpu = resolved_device in ("cuda", "mps")

        # Configure PyTorch for maximum performance BEFORE loading model
        if self._is_gpu:
            _configure_torch_performance(
                enable_tf32=enable_tf32, enable_cudnn_benchmark=enable_cudnn_benchmark
            )

        # Resolve precision
        resolved_precision = self._resolve_precision(precision, resolved_device)
        self._precision = resolved_precision

        # Try to apply power limit BEFORE loading model (reduces load spike)
        if self._is_gpu and gpu_power_limit_watts:
            _try_apply_power_limit(gpu_power_limit_watts)

        # Log model loading - this can cause crashes on GPU memory issues
        diag_log.start_operation(
            "model_load",
            model_name=model_name,
            device=resolved_device,
            precision=resolved_precision,
        )
        logger.info(
            f"Loading embedding model {model_name} on {resolved_device} ({resolved_precision})"
        )

        try:
            import torch

            diag_log.log_pre_cuda_op(
                "SentenceTransformer_init",
                model=model_name,
                device=resolved_device,
                precision=resolved_precision,
            )
            load_start = time.perf_counter()

            # Load model with appropriate precision
            if resolved_precision == "fp16" and self._is_gpu:
                # Load in FP16 for reduced memory and power
                self._model = SentenceTransformer(
                    model_name,
                    device=resolved_device,
                    model_kwargs={"torch_dtype": torch.float16},
                )
                logger.info("Model loaded in FP16 precision (50% less memory, 30% less power)")
            else:
                # Standard FP32 loading
                self._model = SentenceTransformer(model_name, device=resolved_device)

            load_duration = (time.perf_counter() - load_start) * 1000
            diag_log.log_post_cuda_op("SentenceTransformer_init", duration_ms=load_duration)
            diag_log.end_operation("model_load", success=True)
            logger.info(f"Model loaded in {load_duration:.0f}ms")
        except Exception as e:
            diag_log.end_operation("model_load", success=False, error=str(e))
            logger.error(f"Failed to load model: {e}")
            raise

        # Warm up GPU to bring it out of deep idle state
        if self._is_gpu and enable_gpu_warmup:
            _warmup_gpu(self._model)

        self._batch_size = batch_size
        self._model_name = model_name

        # GPU safeguard settings
        self._enable_gpu_safeguards = enable_gpu_safeguards
        self._max_memory_percent = max_memory_percent
        self._max_temperature_c = max_temperature_c
        self._inter_batch_delay = inter_batch_delay
        self._adaptive_batch_size = adaptive_batch_size
        self._min_batch_size = min_batch_size

        # Only enable safeguards for GPU devices
        if self._is_gpu and enable_gpu_safeguards:
            logger.info(
                f"GPU safeguards enabled: max_memory={max_memory_percent}%, "
                f"max_temp={max_temperature_c}°C, adaptive_batch={adaptive_batch_size}"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string to actual device.

        Args:
            device: Device specification ("auto", "cpu", "cuda", "mps").

        Returns:
            Resolved device string.
        """
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
        """Resolve precision based on device capabilities.

        Args:
            precision: Requested precision ("auto", "fp16", "fp32").
            device: Resolved device string.

        Returns:
            Resolved precision ("fp16" or "fp32").
        """
        if precision != "auto":
            return precision

        # Auto-detect: Use FP16 on capable GPUs
        if device == "cpu":
            return "fp32"  # FP16 on CPU is usually slower

        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                # Check compute capability (FP16 good on CC >= 5.3)
                cc = torch.cuda.get_device_capability(0)
                if cc[0] > 5 or (cc[0] == 5 and cc[1] >= 3):
                    return "fp16"
            elif device == "mps":
                # Apple Silicon supports FP16 well
                return "fp16"
        except Exception:
            pass

        return "fp32"

    def embed_documents(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple documents with GPU safeguards.

        Args:
            texts: List of document texts to embed.
            progress_callback: Optional callback(processed, total) for progress updates.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        # Calculate character count for stats
        char_count = sum(len(t) for t in texts)
        start_time = time.perf_counter()
        success = True

        try:
            # Use GPU-safe batching if enabled and on GPU
            if self._is_gpu and self._enable_gpu_safeguards:
                return self._embed_documents_safe(texts, progress_callback)

            # Standard embedding without safeguards (CPU or disabled)
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)
        except Exception:
            success = False
            raise
        finally:
            # Record stats
            duration_ms = (time.perf_counter() - start_time) * 1000
            stats = get_stats_collector()
            stats.record_embedding(
                duration_ms=duration_ms,
                num_texts=len(texts),
                batch_size=self._batch_size,
                char_count=char_count,
                success=success,
            )

    def _embed_documents_safe(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NDArray[np.float32]:
        """Embed documents with GPU memory safeguards.

        Uses throttled batch processing to prevent GPU memory exhaustion
        and thermal issues.

        Args:
            texts: List of document texts to embed.
            progress_callback: Optional progress callback.

        Returns:
            Array of embeddings.
        """
        from rag_service.core.gpu_utils import (
            clear_gpu_memory,
            get_gpu_status,
            throttled_batch_processor,
        )

        logger.info(f"Embedding {len(texts)} documents with GPU safeguards")

        # Clear memory before starting
        clear_gpu_memory()

        # Log initial GPU state
        status = get_gpu_status()
        if status.available:
            logger.info(
                f"GPU: {status.device_name}, "
                f"Memory: {status.memory_used_gb:.1f}/{status.memory_total_gb:.1f}GB "
                f"({status.memory_percent:.1f}%)"
                + (f", Temp: {status.temperature_c}°C" if status.temperature_c else "")
            )

        def encode_batch(batch: list[str]) -> list[NDArray[np.float32]]:
            """Encode a single batch."""
            result = self._model.encode(
                batch,
                batch_size=len(batch),  # Process as single batch
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return [result[i].astype(np.float32) for i in range(len(batch))]

        # Use throttled processor
        embedding_list = throttled_batch_processor(
            items=texts,
            processor=encode_batch,
            batch_size=self._batch_size,
            max_memory_percent=self._max_memory_percent,
            max_temperature_c=self._max_temperature_c,
            inter_batch_delay=self._inter_batch_delay,
            adaptive_batch_size=self._adaptive_batch_size,
            min_batch_size=self._min_batch_size,
            progress_callback=progress_callback,
        )

        # Clear memory after completion
        clear_gpu_memory()

        return np.array(embedding_list, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding array with shape (embedding_dim,).
        """
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        dim = self._model.get_sentence_embedding_dimension()
        return dim if dim is not None else 384  # Default to common embedding size

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def get_device_info(self) -> dict[str, str]:
        """Get information about the device being used.

        Returns:
            Dictionary with device information.
        """
        device = str(self._model.device)
        info = {"device": device, "model": self._model_name}

        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = (
                    f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
                )
        except Exception:
            pass

        return info


def create_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
    batch_size: int = 32,
    # GPU safeguard settings
    enable_gpu_safeguards: bool = True,
    max_memory_percent: float = 80.0,
    max_temperature_c: float = 75.0,
    inter_batch_delay: float = 0.05,
    adaptive_batch_size: bool = True,
    min_batch_size: int = 4,
    # Power management settings
    gpu_power_limit_watts: int | None = None,
    enable_gpu_warmup: bool = True,
    # Precision settings
    precision: PrecisionType = "auto",
    # Performance settings
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
) -> SentenceTransformerEmbedding:
    """Factory function to create an embedding service.

    Args:
        model_name: HuggingFace model ID for embeddings.
        device: Device to run inference on.
        batch_size: Batch size for embedding generation.
        enable_gpu_safeguards: Enable GPU memory/temperature monitoring.
        max_memory_percent: Maximum GPU memory usage before throttling.
        max_temperature_c: Maximum GPU temperature before throttling.
        inter_batch_delay: Seconds to pause between batches.
        adaptive_batch_size: Automatically reduce batch size under pressure.
        min_batch_size: Minimum batch size when adapting.
        gpu_power_limit_watts: Target GPU power limit (requires admin). None to skip.
        enable_gpu_warmup: Warm up GPU after loading to reduce power spikes.
        precision: Floating point precision ("fp32", "fp16", "auto").
        enable_tf32: Enable TF32 on Ampere+ GPUs (~3x faster).
        enable_cudnn_benchmark: Enable cuDNN benchmark mode.

    Returns:
        Configured embedding service instance.
    """
    return SentenceTransformerEmbedding(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        enable_gpu_safeguards=enable_gpu_safeguards,
        max_memory_percent=max_memory_percent,
        max_temperature_c=max_temperature_c,
        inter_batch_delay=inter_batch_delay,
        adaptive_batch_size=adaptive_batch_size,
        min_batch_size=min_batch_size,
        gpu_power_limit_watts=gpu_power_limit_watts,
        enable_gpu_warmup=enable_gpu_warmup,
        precision=precision,
        enable_tf32=enable_tf32,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
    )

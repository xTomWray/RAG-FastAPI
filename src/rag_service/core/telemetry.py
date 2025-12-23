"""OpenTelemetry instrumentation for the RAG Documentation Service.

Provides comprehensive tracing, metrics, and diagnostics for:
- HTTP request/response lifecycle
- Document chunking and processing
- Embedding generation (with GPU metrics)
- Vector store operations
- Graph store operations
- Query routing and execution

Usage:
    from rag_service.core.telemetry import setup_telemetry, traced

    # Initialize on startup
    setup_telemetry()

    # Decorate functions to trace
    @traced("embedding.encode")
    def encode_texts(texts: list[str]) -> np.ndarray:
        ...

    # Or use context manager for fine-grained control
    with tracer.start_as_current_span("custom_operation") as span:
        span.set_attribute("custom.key", "value")
        ...
"""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

# Type hints for decorators
P = ParamSpec("P")
R = TypeVar("R")

# Global state for telemetry
_telemetry_initialized = False
_tracer = None
_meter = None

# Try to import OpenTelemetry - gracefully handle if not installed
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import Span, TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]
    Span = None  # type: ignore[misc, assignment]


def is_telemetry_available() -> bool:
    """Check if OpenTelemetry is installed and available."""
    return OTEL_AVAILABLE


def is_telemetry_enabled() -> bool:
    """Check if telemetry is both available and enabled in config."""
    if not OTEL_AVAILABLE:
        return False
    from rag_service.config import get_settings

    return get_settings().enable_telemetry


def setup_telemetry() -> bool:
    """Initialize OpenTelemetry tracing and metrics.

    Configures the tracer and meter providers based on settings.
    Safe to call multiple times - subsequent calls are no-ops.

    Returns:
        True if telemetry was initialized, False if skipped or unavailable.
    """
    global _telemetry_initialized, _tracer, _meter

    if _telemetry_initialized:
        return True

    if not OTEL_AVAILABLE:
        logger.info("OpenTelemetry not installed. Install with: pip install .[observability]")
        return False

    from rag_service.config import get_settings

    settings = get_settings()

    if not settings.enable_telemetry:
        logger.debug("Telemetry disabled in configuration")
        return False

    try:
        # Create resource with service info
        resource = Resource.create(
            {
                SERVICE_NAME: settings.telemetry_service_name,
                "service.version": "0.1.0",
                "deployment.environment": "development",
            }
        )

        # Configure sampling
        sampler = TraceIdRatioBased(settings.telemetry_sample_rate)

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource, sampler=sampler)

        # Configure exporter based on settings
        if settings.telemetry_exporter == "console":
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            tracer_provider.add_span_processor(processor)
            logger.info("Telemetry configured with console exporter")

        elif settings.telemetry_exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=settings.telemetry_endpoint)
                otlp_processor = BatchSpanProcessor(exporter)
                tracer_provider.add_span_processor(otlp_processor)
                logger.info(
                    f"Telemetry configured with OTLP exporter: {settings.telemetry_endpoint}"
                )
            except ImportError:
                logger.warning("OTLP exporter not available. Install opentelemetry-exporter-otlp")
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                tracer_provider.add_span_processor(processor)

        elif settings.telemetry_exporter == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                jaeger_exporter = JaegerExporter(
                    agent_host_name=settings.telemetry_endpoint.split(":")[0].replace(
                        "http://", ""
                    ),
                    agent_port=int(settings.telemetry_endpoint.split(":")[-1])
                    if ":" in settings.telemetry_endpoint
                    else 6831,
                )
                jaeger_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(jaeger_processor)
                logger.info(
                    f"Telemetry configured with Jaeger exporter: {settings.telemetry_endpoint}"
                )
            except ImportError:
                logger.warning(
                    "Jaeger exporter not available. Install opentelemetry-exporter-jaeger"
                )
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                tracer_provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Create meter provider for metrics
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)

        # Get tracer and meter instances
        _tracer = trace.get_tracer(__name__, "0.1.0")
        _meter = metrics.get_meter(__name__, "0.1.0")

        _telemetry_initialized = True
        logger.info(
            f"Telemetry initialized: service={settings.telemetry_service_name}, "
            f"exporter={settings.telemetry_exporter}, sample_rate={settings.telemetry_sample_rate}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        return False


def get_tracer() -> Any:
    """Get the configured tracer instance.

    Returns a no-op tracer if telemetry is not enabled.
    """
    global _tracer
    if _tracer is None:
        if OTEL_AVAILABLE:
            return trace.get_tracer(__name__)
        return NoOpTracer()
    return _tracer


def get_meter() -> Any:
    """Get the configured meter instance.

    Returns a no-op meter if telemetry is not enabled.
    """
    global _meter
    if _meter is None:
        if OTEL_AVAILABLE:
            return metrics.get_meter(__name__)
        return NoOpMeter()
    return _meter


class NoOpSpan:
    """No-operation span for when telemetry is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        pass


class NoOpTracer:
    """No-operation tracer for when telemetry is disabled."""

    def start_span(self, _name: str, **_kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def start_as_current_span(self, _name: str, **_kwargs: Any) -> NoOpSpan:
        return NoOpSpan()


class NoOpMeter:
    """No-operation meter for when telemetry is disabled."""

    def create_counter(self, _name: str, **_kwargs: Any) -> "NoOpCounter":
        return NoOpCounter()

    def create_histogram(self, _name: str, **_kwargs: Any) -> "NoOpHistogram":
        return NoOpHistogram()

    def create_gauge(self, _name: str, **_kwargs: Any) -> "NoOpGauge":
        return NoOpGauge()


class NoOpCounter:
    def add(self, value: int, attributes: dict[str, Any] | None = None) -> None:
        pass


class NoOpHistogram:
    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        pass


class NoOpGauge:
    def set(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        pass


def traced(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
    record_args: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to trace a function execution.

    Creates a span around the function call with timing and optional
    argument recording.

    Args:
        span_name: Name for the span (defaults to function name).
        attributes: Static attributes to add to the span.
        record_exception: Whether to record exceptions in the span.
        record_args: Whether to record function arguments as attributes.

    Example:
        @traced("embedding.encode", attributes={"model": "all-MiniLM-L6-v2"})
        def encode_texts(texts: list[str]) -> np.ndarray:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(name) as span:
                # Add static attributes
                if attributes:
                    span.set_attributes(attributes)

                # Add function arguments if requested
                if record_args:
                    for i, arg in enumerate(args):
                        span.set_attribute(f"arg.{i}", _safe_str(arg))
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg.{key}", _safe_str(value))

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

                    if record_exception:
                        span.record_exception(e)
                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.ERROR, str(e)))

                    raise

        return wrapper

    return decorator


def traced_async(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Async version of the traced decorator.

    Example:
        @traced_async("query.search")
        async def search(query: str) -> list[Document]:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(name) as span:
                if attributes:
                    span.set_attributes(attributes)

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result  # type: ignore[no-any-return]

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

                    if record_exception:
                        span.record_exception(e)
                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.ERROR, str(e)))

                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Context manager for creating spans with automatic timing.

    Example:
        with span("vector_store.search", {"collection": "documents"}) as s:
            results = store.search(query)
            s.set_attribute("results.count", len(results))
    """
    tracer = get_tracer()
    start_time = time.perf_counter()

    with tracer.start_as_current_span(name) as s:
        if attributes:
            s.set_attributes(attributes)

        try:
            yield s
            duration_ms = (time.perf_counter() - start_time) * 1000
            s.set_attribute("duration_ms", duration_ms)
            if OTEL_AVAILABLE:
                s.set_status(Status(StatusCode.OK))
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            s.set_attribute("duration_ms", duration_ms)
            s.record_exception(e)
            if OTEL_AVAILABLE:
                s.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def add_gpu_metrics(span_obj: Any) -> None:
    """Add GPU metrics as span attributes.

    Captures current GPU state including:
    - Memory usage (used, total, percent)
    - Temperature
    - Utilization
    - Power draw

    Args:
        span_obj: The span to add attributes to.
    """
    from rag_service.config import get_settings

    settings = get_settings()

    if not settings.telemetry_include_gpu_metrics:
        return

    try:
        import torch

        if not torch.cuda.is_available():
            span_obj.set_attribute("gpu.available", False)
            return

        span_obj.set_attribute("gpu.available", True)
        span_obj.set_attribute("gpu.device_count", torch.cuda.device_count())
        span_obj.set_attribute("gpu.current_device", torch.cuda.current_device())
        span_obj.set_attribute("gpu.device_name", torch.cuda.get_device_name(0))

        # Memory metrics
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        span_obj.set_attribute("gpu.memory.allocated_gb", round(memory_allocated, 2))
        span_obj.set_attribute("gpu.memory.reserved_gb", round(memory_reserved, 2))
        span_obj.set_attribute("gpu.memory.total_gb", round(memory_total, 2))
        span_obj.set_attribute(
            "gpu.memory.percent", round(memory_allocated / memory_total * 100, 1)
        )

        # Try to get nvidia-smi metrics
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    span_obj.set_attribute("gpu.temperature_c", float(parts[0]))
                    span_obj.set_attribute("gpu.utilization_percent", float(parts[1]))
                    span_obj.set_attribute("gpu.power_watts", float(parts[2]))
        except Exception:
            pass  # nvidia-smi not available

    except ImportError:
        span_obj.set_attribute("gpu.available", False)
    except Exception as e:
        span_obj.set_attribute("gpu.metrics_error", str(e))


def add_document_metrics(span_obj: Any, documents: list[Any]) -> None:
    """Add document-related metrics to a span.

    Args:
        span_obj: The span to add attributes to.
        documents: List of documents being processed.
    """
    if not documents:
        span_obj.set_attribute("documents.count", 0)
        return

    span_obj.set_attribute("documents.count", len(documents))

    # Calculate text statistics
    total_chars = sum(len(getattr(doc, "text", str(doc))) for doc in documents)
    avg_chars = total_chars / len(documents) if documents else 0

    span_obj.set_attribute("documents.total_chars", total_chars)
    span_obj.set_attribute("documents.avg_chars", round(avg_chars, 1))

    # Count unique sources if available
    sources = set()
    for doc in documents:
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = doc.metadata.get("source")
            if source:
                sources.add(source)

    if sources:
        span_obj.set_attribute("documents.unique_sources", len(sources))


def add_embedding_metrics(span_obj: Any, texts: list[str], embeddings: Any = None) -> None:
    """Add embedding-related metrics to a span.

    Args:
        span_obj: The span to add attributes to.
        texts: List of texts being embedded.
        embeddings: Optional resulting embeddings array.
    """
    span_obj.set_attribute("embedding.text_count", len(texts))

    if texts:
        total_chars = sum(len(t) for t in texts)
        span_obj.set_attribute("embedding.total_chars", total_chars)
        span_obj.set_attribute("embedding.avg_chars", round(total_chars / len(texts), 1))

    if embeddings is not None:
        try:
            span_obj.set_attribute("embedding.output_shape", str(embeddings.shape))
            span_obj.set_attribute("embedding.dimension", embeddings.shape[-1])
        except Exception:
            pass


def add_search_metrics(
    span_obj: Any, query: str, results: list[Any], scores: list[float] | None = None
) -> None:
    """Add search-related metrics to a span.

    Args:
        span_obj: The span to add attributes to.
        query: The search query.
        results: Search results.
        scores: Optional relevance scores.
    """
    span_obj.set_attribute("search.query_length", len(query))
    span_obj.set_attribute("search.results_count", len(results))

    if scores:
        span_obj.set_attribute("search.max_score", max(scores))
        span_obj.set_attribute("search.min_score", min(scores))
        span_obj.set_attribute("search.avg_score", round(sum(scores) / len(scores), 4))


def _safe_str(value: Any, max_length: int = 200) -> str:
    """Convert a value to a safe string representation for span attributes."""
    try:
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s
    except Exception:
        return "<unable to convert>"


# Create standard metrics
def create_standard_metrics() -> dict[str, Any]:
    """Create standard metrics for the RAG service."""
    meter = get_meter()

    return {
        # Request metrics
        "request_count": meter.create_counter(
            "rag.requests.total",
            description="Total number of requests",
        ),
        "request_duration": meter.create_histogram(
            "rag.requests.duration_ms",
            description="Request duration in milliseconds",
        ),
        # Embedding metrics
        "embedding_count": meter.create_counter(
            "rag.embeddings.total",
            description="Total number of embedding operations",
        ),
        "embedding_duration": meter.create_histogram(
            "rag.embeddings.duration_ms",
            description="Embedding generation duration in milliseconds",
        ),
        "embedding_batch_size": meter.create_histogram(
            "rag.embeddings.batch_size",
            description="Batch sizes for embedding operations",
        ),
        # Document metrics
        "documents_processed": meter.create_counter(
            "rag.documents.processed_total",
            description="Total documents processed",
        ),
        "chunks_created": meter.create_counter(
            "rag.chunks.created_total",
            description="Total chunks created",
        ),
        # Search metrics
        "search_count": meter.create_counter(
            "rag.search.total",
            description="Total search operations",
        ),
        "search_duration": meter.create_histogram(
            "rag.search.duration_ms",
            description="Search duration in milliseconds",
        ),
        # GPU metrics
        "gpu_memory_used": meter.create_gauge(
            "rag.gpu.memory_used_gb",
            description="GPU memory used in GB",
        ),
        "gpu_temperature": meter.create_gauge(
            "rag.gpu.temperature_c",
            description="GPU temperature in Celsius",
        ),
    }


def instrument_fastapi(app: Any) -> None:
    """Add OpenTelemetry instrumentation to a FastAPI app.

    This automatically traces all HTTP requests with:
    - Request method and path
    - Response status code
    - Request duration
    - Exception details (if any)

    Args:
        app: FastAPI application instance.
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        logger.debug("Skipping FastAPI instrumentation (telemetry disabled)")
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")

    except ImportError:
        logger.warning(
            "FastAPI instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-fastapi"
        )
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")

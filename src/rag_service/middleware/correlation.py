"""Correlation ID middleware for request tracing.

Generates or propagates correlation IDs for distributed tracing.
The correlation ID is bound to the structlog context and included
in response headers.

Example:
    # Incoming request with correlation ID
    GET /api/v1/query HTTP/1.1
    X-Correlation-ID: abc-123-def

    # Response includes same ID
    HTTP/1.1 200 OK
    X-Correlation-ID: abc-123-def

    # All logs during request include correlation_id
    {"event": "Processing query", "correlation_id": "abc-123-def", ...}
"""

import uuid
from contextvars import ContextVar
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

import structlog

# Header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"

# Context variable for storing correlation ID within request scope
_correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get the current correlation ID from context.

    Returns:
        The correlation ID for the current request, or empty string if not set.
    """
    return _correlation_id_ctx.get()


def _generate_correlation_id() -> str:
    """Generate a new correlation ID.

    Returns:
        A new UUID4 string for use as a correlation ID.
    """
    return str(uuid.uuid4())


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that manages correlation IDs for request tracing.

    For each incoming request:
    1. Extracts correlation ID from X-Correlation-ID header, or generates a new one
    2. Stores it in a context variable for access during request handling
    3. Binds it to structlog context for automatic inclusion in all logs
    4. Adds the correlation ID to the response headers

    This enables end-to-end request tracing across distributed systems.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application to wrap.
        """
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request, managing correlation ID lifecycle.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response with correlation ID header added.
        """
        # Get existing correlation ID or generate a new one
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        if not correlation_id:
            correlation_id = _generate_correlation_id()

        # Store in context variable
        token = _correlation_id_ctx.set(correlation_id)

        try:
            # Bind to structlog context
            structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

            # Process the request
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers[CORRELATION_ID_HEADER] = correlation_id

            return response
        finally:
            # Reset context variable
            _correlation_id_ctx.reset(token)
            # Clear structlog context
            structlog.contextvars.clear_contextvars()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from prometheus.exceptions.server_exception import ServerException


def register_exception_handlers(app: FastAPI):
    """Global exception handlers for the FastAPI application."""

    @app.exception_handler(ServerException)
    async def custom_exception_handler(_request: Request, exc: ServerException):
        """
        Custom exception handler for ServerException.
        """
        return JSONResponse(
            status_code=exc.code, content={"code": exc.code, "message": exc.message, "data": None}
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, _exc: Exception):
        """
        Global exception handler for all uncaught exceptions.
        """
        return JSONResponse(
            status_code=500, content={"code": 500, "message": "Internal Server Error", "data": None}
        )

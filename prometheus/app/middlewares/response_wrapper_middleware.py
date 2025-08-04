from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import json


class ResponseWrapperMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Check if the response is a JSON response
        if response.headers.get("content-type") == "application/json":
            # Decode the JSON response body
            original_data = json.loads(response.body.decode("utf-8"))

            # Check if the original data is already in the expected format
            if isinstance(original_data, dict) and set(original_data.keys()) == {"code", "message", "data"}:
                new_data = original_data
            else:
                new_data = {
                    "code": response.status_code,
                    "message": "Success",
                    "data": original_data,
                }
            return JSONResponse(content=new_data, status_code=response.status_code)
        return response

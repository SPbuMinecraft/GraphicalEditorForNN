from typing import Any
from http import HTTPStatus
from werkzeug.exceptions import HTTPException
from requests.exceptions import ConnectionError, ConnectTimeout


class Error(Exception):
    """Generic class to send back any kind of error
    Example:
        raise Error("You did bad things", HTTPStatus.FORBIDDEN, description="some", data="more")
    In any place in your code
    """

    def __init__(
        self, message: str, status_code: int = HTTPStatus.BAD_REQUEST, **kwargs: Any
    ):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload: dict[str, Any] = kwargs

    def __str__(self) -> str:
        return self.message

    def to_dict(self):
        result = self.payload or {}
        result["error"] = self.message
        return result


def api_error(e: Error):
    return e.to_dict(), e.status_code


def key_error(e: KeyError):
    return {"error": f"Missing required field {str(e)}"}, HTTPStatus.BAD_REQUEST


def value_error(e: ValueError):
    return {"error": f"Bad format: {str(e)}"}, HTTPStatus.BAD_REQUEST


def http_error(e: HTTPException):
    return {"error": str(e)}, HTTPStatus.BAD_REQUEST


def timeout_error(e: ConnectTimeout):
    return {"error": "Request to c++ server timeout"}, HTTPStatus.INTERNAL_SERVER_ERROR


def connection_error(e: ConnectionError):
    return {
        "error": "Failed to connect to the c++ server"
    }, HTTPStatus.INTERNAL_SERVER_ERROR

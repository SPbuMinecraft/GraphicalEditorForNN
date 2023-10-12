import re
import typing as tp
from enum import Enum

from flask import Response, abort, current_app


class LayersConnectionStatus(Enum):
    OK = 0
    DoNotExist = 1
    AccessDenied = 2
    DimensionsMismatch = 3


class DeleteStatus(Enum):
    OK = 0
    ModelNotExist = 1
    ElementNotExist = 2
    LayerNotFree = 3


def error(code: int, message: str):
    abort(Response(message, code))


def get_cpp_server_address():
    cpp = current_app.config["cpp_server"]
    return f"http://{cpp['PORT']}:{cpp['PORT']}"


def is_valid_model(model_dict):
    return True


def parse_parameters(layer_string: str) -> dict[str, tp.Any]:
    params_dict = {}
    for param in layer_string.split(";"):
        param = param.strip()
        param_name, param_value = param.split("=")
        if re.match(r"^\[[^,]+(,[^,]+)*\]$", param_value) is not None:
            params_dict[param_name] = list(param_value[1:-1].split(","))
        else:
            params_dict[param_name] = param_value
    return params_dict

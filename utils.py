import re
import typing as tp
import os
from enum import Enum

from flask import Response, abort


class LayersConnectionStatus(Enum):
    OK = 0
    DoNotExist = 1
    AccessDenied = 2
    DimensionsMismatch = 3


def to_port(num: int) -> int:
    return int(os.environ.get("PORT", num))


def error(code: int, message: str):
    abort(Response(message, code))


def is_valid_model(model_dict):
    return True


def parse_parameters(layer_string: str) -> dict[str, tp.Any]:
    params_dict = {}
    for param in layer_string.split(';'):
        param = param.strip()
        param_name, param_value = param.split('=')
        if re.match(r'^\[[^,]+(,[^,]+)*\]$', param_value) is not None:
            params_dict[param_name] = list(param_value[1:-1].split(','))
        else:
            params_dict[param_name] = param_value
    return params_dict

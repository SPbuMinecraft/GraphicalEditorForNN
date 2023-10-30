import re
import typing as tp
from enum import Enum
from collections import deque

from flask import Response, abort


class LayersConnectionStatus(Enum):
    OK = 0
    DoNotExist = 1
    AccessDenied = 2
    DimensionsMismatch = 3
    WrongDirection = 4
    Cycle = 5


class DeleteStatus(Enum):
    OK = 0
    ModelNotExist = 1
    ElementNotExist = 2
    LayerNotFree = 3


def error(code: int, message: str) -> tp.NoReturn:
    abort(Response(message, code))


def get_edges_from_model(model_dict):
    edges = dict()
    for connection in model_dict["connections"]:
        layer_from, layer_to = connection["layer_from"], connection["layer_to"]
        if layer_from in edges.keys():
            edges[layer_from].append(layer_to)
        else:
            edges[layer_from] = [layer_to]
    return edges


def is_valid_model(model_dict):
    start_candidates = list(
        filter(lambda x: x["layer_type"] == "Data", model_dict["layers"])
    )
    stop_candidates = list(
        filter(lambda x: x["layer_type"] == "Output", model_dict["layers"])
    )
    if len(start_candidates) == 0 or len(stop_candidates) == 0:
        return False
    start_ids = list(map(lambda layer: layer["id"], start_candidates))
    stop_ids = list(map(lambda layer: layer["id"], stop_candidates))
    return check_paths_exist(start_ids, stop_ids, model_dict)


def check_paths_exist(start_ids: list, stop_ids: list, model_dict: dict):
    edges = get_edges_from_model(model_dict)
    # BFS realisation
    distance = dict()
    for layer in model_dict["layers"]:
        distance[layer["id"]] = 0 if layer["id"] in start_ids else -1
    layers_queue = deque(start_ids)
    while layers_queue:
        layer = layers_queue[0]
        layers_queue.popleft()
        for layer_to in edges.get(layer, []):
            if distance[layer_to] == -1:
                distance[layer_to] = distance[layer] + 1
                layers_queue.append(layer_to)
    return all([distance[stop] != -1 for stop in stop_ids])


def parse_parameters(layer_string: str) -> dict[str, tp.Any]:
    params_dict = {}
    for param in layer_string.split(";"):
        param = param.strip()
        if not param:
            continue
        param_name, param_value = param.split("=")
        param_name = param_name.strip()
        param_value = param_value.strip()
        if re.match(r"^\[[^,]+(,[^,]+)*\]$", param_value) is not None:
            params_dict[param_name] = list(param_value[1:-1].split(","))
        else:
            params_dict[param_name] = param_value  # type: ignore
    return params_dict

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


class DeleteStatus(Enum):
    OK = 0
    ModelNotExist = 1
    ElementNotExist = 2
    LayerNotFree = 3


def to_port(num: int) -> int:
    return int(os.environ.get("PORT", num))


def error(code: int, message: str):
    abort(Response(message, code))


def get_edges_from_model(model_dict):
    edges = dict()
    for connection in model_dict['connections']:
        layer_from, layer_to = connection['layer_from'], connection['layer_to']
        if layer_from in edges.keys():
            edges[layer_from].append(layer_to)
        else:
            edges[layer_from] = [layer_to]
    return edges


def is_valid_model(model_dict):
    edges = get_edges_from_model(model_dict)
    start_candidates = list(filter(lambda x: x['type'] == 'Data', model_dict['layers']))
    stop_candidates = list(filter(lambda x: x['type'] == 'Output', model_dict['layers']))
    if len(start_candidates) != 1 or len(stop_candidates) != 1:
        return False
    start = start_candidates[0]['id']
    stop = stop_candidates[0]['id']
    # BFS realisation
    distance = dict()
    for layer in model_dict['layers']:
        distance[layer['id']] = 0 if layer['id'] == start else -1
    layers_queue = [start]
    while layers_queue:
        layer = layers_queue[0]
        layers_queue.pop(0)
        for layer_to in edges[layer]:
            if distance[layer_to] == -1:
                distance[layer_to] = distance[layer] + 1
                layers_queue.append(layer_to)
    return distance[stop] != -1


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

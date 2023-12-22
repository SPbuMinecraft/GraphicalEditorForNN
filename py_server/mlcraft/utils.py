import re
import os
import datetime
import typing as tp
from enum import Enum
from collections import deque, defaultdict
import matplotlib.pyplot as plt

plt.switch_backend("Agg")


def get_edges_from_model(model_dict):
    edges = defaultdict(list)
    for layer_to in model_dict["layers"]:
        for layer_from in layer_to["parents"]:
            edges[layer_from].append(layer_to["id"])
    return edges


def is_valid_model(model_dict):
    start_candidates = list(filter(lambda x: x["type"] == "Data", model_dict["layers"]))
    stop_candidates = list(
        filter(lambda x: x["type"] == "Output", model_dict["layers"])
    )
    # Проверяем соединятеся ли data и loss
    # TODO: MSELoss -> Loss
    loss_layer = list(filter(lambda x: x["type"] == "MSELoss", model_dict["layers"]))
    target_layer = list(filter(lambda x: x["type"] == "Target", model_dict["layers"]))

    if len(start_candidates) == 0 or len(stop_candidates) == 0:
        return False
    start_ids = list(map(lambda layer: layer["id"], start_candidates))
    stop_ids = list(map(lambda layer: layer["id"], stop_candidates))
    loss_ids = list(map(lambda layer: layer["id"], loss_layer))
    target_ids = list(map(lambda layer: layer["id"], target_layer))
    return (
        check_paths_exist(start_ids, stop_ids, model_dict)
        and check_paths_exist(start_ids, loss_ids, model_dict)
        and check_paths_exist(target_ids, loss_ids, model_dict)
    )


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


def topology_sort(entry_nodes: list[int], edges: dict[int, list[int]]) -> list[int]:
    closed = set()
    dfs_stack = []
    is_final = False
    final_order = []

    for entry_node in entry_nodes:
        dfs_stack.append(entry_node)
        while dfs_stack:
            current_node = dfs_stack[-1]
            is_final = True
            if current_node in edges:
                for next_node in edges[current_node]:
                    if next_node in closed:
                        continue
                    is_final = False
                    dfs_stack.append(next_node)
            if is_final:
                final_order.append(current_node)
                closed.add(current_node)
                dfs_stack.pop()
    return list(reversed(final_order))


def convert_model_parameters(model):
    """Convert from frontend style parameters to json style"""

    for i in range(len(model["layers"])):
        model["layers"][i]["parameters"] = parse_parameters(
            model["layers"][i]["parameters"]
        )


def convert_model(model):
    """Convert json to another format for C++ by deleting connsetcions ids and rename layers_type"""

    model["connections"] = list(
        {
            "layer_from": layer_from,
            "layer_to": layer_to["id"],
        }
        for layer_to in model["layers"]
        for layer_from in layer_to["parents"]
    )  # Create list of connections for C++ server

    model["layers"] = list(
        map(
            lambda layer: {
                "id": layer["id"],
                "type": layer["type"],
                "parameters": layer["parameters"],
            },
            model["layers"],
        )
    )


def plot_metrics(values: list[float], user_id: int, model_id: int, label: str) -> str:
    dt = datetime.datetime.now()
    plt.plot(list(range(1, len(values) + 1)), values)
    isoformat_dt = re.sub(r"[:\.\-]", "_", dt.isoformat())
    path = f"{user_id}_{model_id}_{label}_{isoformat_dt}.png"
    plt.savefig(os.path.join("images", path))
    return path


def delete_file(path: str):
    if os.path.exists(path):
        os.remove(path)

from enum import Enum
from collections import defaultdict
import typing as tp

from .utils import topology_sort


class DimensionsCheckStatus(Enum):
    OK = 0
    InvalidNumberOfInputs = 1
    DimensionsMismatch = 2


class Data2dChecker:
    def __init__(self, width):
        self.output_shape = [-1, width]

    def __call__(self):
        return True


class LinearChecker:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, input_shape: list[int]):
        self.output_shape = input_shape.copy()
        if not input_shape or input_shape[-1] != self.in_features:
            return False
        self.output_shape[-1] = self.out_features
        return True


class ReLUChecker:
    def __init__(self):
        pass

    def __call__(self, input_shape: list[int]):
        self.output_shape = input_shape.copy()
        return True


class SumChecker:
    def __init__(self):
        pass

    def __call__(self, first_input_shape: list[int], second_input_shape: list[int]):
        if not first_input_shape or first_input_shape != second_input_shape:
            return False
        self.output_shape = first_input_shape.copy()
        return True


class MSEChecker:
    def __init__(self):
        pass

    def __call__(self, input_shape: list[int], targets_shape: list[int]):
        if not input_shape or input_shape != targets_shape:
            return False
        self.output_shape = input_shape[0]
        return True


def create_checker(layer: dict):
    if layer["type"] in ("Data", "Target"):
        print(layer)
        return Data2dChecker(layer["parameters"]["width"])
    elif layer["type"] == "Linear":
        return LinearChecker(
            layer["parameters"]["inFeatures"], layer["parameters"]["outFeatures"]
        )
    elif layer["type"] == "ReLU":
        return ReLUChecker()
    elif layer["type"] == "MSELoss":
        return MSEChecker()
    elif layer["type"] == "Sum":
        return SumChecker()
    else:
        raise TypeError(f"Unknown layer type: {layer['type']}")


def check_dimensions(
    layers: list[dict],
) -> tuple[DimensionsCheckStatus, tp.Optional[int]]:
    layer_checkers = {}
    data_layers = []

    # TODO: maybe it's possible to optimize memory usage is some nice way?
    edges = defaultdict(list)
    parents = defaultdict(list)

    for layer in layers:
        if layer["type"] == "Output":
            continue
        if layer["type"] in ("Data", "Target"):
            data_layers.append(layer["id"])
        layer_checkers[layer["id"]] = create_checker(layer)
        for prev in layer["parents"]:
            edges[prev].append(layer["id"])
            parents[layer["id"]].append(prev)

    layers_order = topology_sort(data_layers, edges)
    current_layer_id = None
    try:
        for layer_id in layers_order:
            current_layer_id = layer_id
            acceptable = layer_checkers[layer_id](
                *[
                    layer_checkers[parent_id].output_shape
                    for parent_id in parents[layer_id]
                ]
            )
            if not acceptable:
                return DimensionsCheckStatus.DimensionsMismatch, layer_id
    except TypeError as e:
        return DimensionsCheckStatus.InvalidNumberOfInputs, current_layer_id
    return DimensionsCheckStatus.OK, None

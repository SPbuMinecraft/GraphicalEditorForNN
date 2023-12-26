from collections import defaultdict
from http import HTTPStatus

from .utils import topology_sort
from .errors import Error


class DataChecker:
    def __init__(self, shape: list[int]):
        self.output_shape = [1] * (3 - len(shape)) + shape

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


class Conv2dChecker:
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, input_shape: list[int]):
        self.output_shape = input_shape.copy()
        if not input_shape or input_shape[0] != self.in_channels:
            return False
        self.output_shape[0] = self.out_channels
        return True


class ReLUChecker:
    def __init__(self):
        pass

    def __call__(self, input_shape: list[int]):
        self.output_shape = input_shape.copy()
        return True


class LayerNormChecker:
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
        self.output_shape = None
        return True


class CrossEntropyChecker:
    def __init__(self, class_count):
        self.class_count = class_count

    def __call__(self, input_shape: list[int], targets_shape: list[int]):
        for prim_dim in range(2):
            if input_shape[prim_dim] != 1 or targets_shape[prim_dim] != 1:
                return False
        if input_shape[-1] != self.class_count or targets_shape[-1] != 1:
            return False
        self.output_shape = None
        return True


class DimReduceChecker:
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, input_shape: list[int]):
        self.output_shape = input_shape.copy()
        for ax in self.axes:
            self.output_shape[ax - 1] = 1
        return True


class PoolingChecker:
    def __init__(self):
        pass

    def __call__(self, input_shape: list[int]):
        if input_shape[1] % 2 != 0 or input_shape[2] % 2 != 0:
            return False
        self.output_shape = input_shape.copy()
        self.output_shape[1] //= 2
        self.output_shape[2] //= 2
        return True


def create_checker(layer: dict):
    match layer["type"]:
        case "Data" | "Target":
            return DataChecker(list(map(int, layer["parameters"]["shape"])))
        case "Linear":
            return LinearChecker(
                int(layer["parameters"]["inFeatures"]),
                int(layer["parameters"]["outFeatures"]),
            )
        case "ReLU":
            return ReLUChecker()
        case "Sum":
            return SumChecker()
        case "LayerNorm":
            return LayerNormChecker()
        case "Loss":
            if layer["parameters"]["type"] == "MSE":
                return MSEChecker()
            elif layer["parameters"]["type"] == "Entropy":
                return CrossEntropyChecker(int(layer["parameters"]["classCount"]))
            else:
                raise TypeError(f"Unknown loss type: {layer['parameters']['type']}")
        case "Sum":
            return SumChecker()
        case "Conv2D":
            return Conv2dChecker(
                int(layer["parameters"]["inChannels"]),
                int(layer["parameters"]["outChannels"]),
            )
        case "Mean" | "SoftMax":
            return DimReduceChecker(list(map(int, layer["parameters"]["axes"])))
        case "Pooling":
            return PoolingChecker()
        case _:
            raise TypeError(f"Unknown layer type: {layer['type']}")


def assert_dimensions_match(layers: list[dict]):
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
                raise Error(
                    f"Layer {layer_id} does not match with it's parents in dimensions",
                    HTTPStatus.NOT_ACCEPTABLE,
                    problemNode=layer_id,
                )
    except TypeError as e:
        raise Error(
            f"Invalid number of inputs for layer {current_layer_id}",
            HTTPStatus.NOT_ACCEPTABLE,
        )

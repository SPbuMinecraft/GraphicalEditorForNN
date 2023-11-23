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
        return Data2dChecker(layer["width"])
    elif layer["type"] == "Linear":
        return LinearChecker(layer["inFeatures"], layer["outFeatures"])
    elif layer["type"] == "ReLU":
        return ReLUChecker()
    elif layer["type"] == "MSELoss":
        return MSEChecker()

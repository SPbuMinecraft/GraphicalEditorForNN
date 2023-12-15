import csv
import math

from http import HTTPStatus
from io import StringIO
from .errors import Error


def get_layer(type: str, model: dict) -> dict:
    layer = next(filter(lambda l: l["type"] == type, model["layers"]), None)
    if layer is None:
        raise Error(f"No {type} block in model", HTTPStatus.PRECONDITION_FAILED)
    return layer


def extract_train_data(bytes: bytes, model: dict) -> dict:
    data_layer = get_layer("Data", model)
    target_layer = get_layer("Target", model)

    reader = csv.reader(StringIO(bytes.decode()))

    data: list[float] = []
    target: list[float] = []
    dims = list(map(int, data_layer["parameters"]["shape"]))
    dims_total = math.prod(dims)
    for row in reader:
        if dims_total != len(row) - 1:  # columns = features + 1 for target
            raise Error("Input csv column count doesn't match data's feature count")

        data.extend(map(float, row[:-1]))
        target.append(float(row[-1]))

    return {data_layer["id"]: data, target_layer["id"]: target}


def extract_predict_data(bytes: bytes, model: dict) -> dict:
    data_layer = get_layer("Data", model)
    reader = csv.reader(StringIO(bytes.decode()))
    row = next(reader, None)

    dims = list(map(int, data_layer["parameters"]["shape"]))
    dims_total = math.prod(dims)
    if row is None:
        raise Error("Bad csv format")
    if dims_total != len(row):  # predict request - one row with data only
        raise Error("Input csv column count doesn't match data's feature count")

    return {data_layer["id"]: list(map(float, row))}

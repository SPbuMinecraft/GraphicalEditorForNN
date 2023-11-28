import csv
from http import HTTPStatus
from io import StringIO
from .db import sql_worker
from .errors import Error


def get_data_layer(model: dict) -> dict:
    layer = next(filter(lambda l: l["type"] == "Data", model["layers"]))
    if layer is None:
        raise Error("No 'data' block in model", HTTPStatus.PRECONDITION_FAILED)
    return layer


def get_target_layer(model: dict) -> dict:
    layer = next(filter(lambda l: l["type"] == "Target", model["layers"]), None)
    if layer is None:
        raise Error("No 'target' block in model", HTTPStatus.PRECONDITION_FAILED)
    return layer


def extract_train_data(bytes: bytes, model: dict) -> dict:
    data_layer = get_data_layer(model)
    target_layer = get_target_layer(model)

    reader = csv.reader(StringIO(bytes.decode()))

    data: list[float] = []
    target: list[float] = []
    for row in reader:
        if (
            int(data_layer["parameters"]["width"]) != len(row) - 1
        ):  # columns = features + 1 for target
            raise Error("Input csv column count doesn't match data's feature count")

        data.extend(map(float, row[:-1]))
        target.append(float(row[-1]))

    return {data_layer["id"]: data, target_layer["id"]: target}


def extract_predict_data(bytes: bytes, model: dict) -> dict:
    data_layer = get_data_layer(model)
    reader = csv.reader(StringIO(bytes.decode()))
    row = next(reader, None)

    if row is None:
        raise Error("Bad csv format")
    if int(data_layer["parameters"]["width"]) != len(
        row
    ):  # predict request - one row with data only
        raise Error("Input csv column count doesn't match data's feature count")

    return {data_layer["id"]: list(map(float, row))}

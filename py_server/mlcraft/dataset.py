import csv
from io import StringIO
from .db import sql_worker


def get_data_id(model_id: int) -> int:
    # TODO: think about asynchronous requests
    items = sql_worker.get_graph_elements(model_id)
    layer = next(filter(lambda l: l["type"] == "Data", items["layers"]))
    return layer["id"]


def get_target_id(model_id: int) -> int:
    items = sql_worker.get_graph_elements(model_id)
    layer = next(filter(lambda l: l["type"] == "Target", items["layers"]))
    return layer["id"]


def extract_train_data(bytes: bytes, model_id: int) -> dict:
    # TODO: when we define formats, errors will be handled
    data_id = get_data_id(model_id)
    target_id = get_target_id(model_id)

    reader = csv.reader(StringIO(bytes.decode()))

    data: list[float] = []
    target: list[float] = []
    for row in reader:
        data.extend(map(float, row[:-1]))
        target.append(float(row[-1]))

    return {data_id: data, target_id: target}


def extract_predict_data(bytes: bytes, model_id: int) -> dict:
    data_id = get_data_id(model_id)
    reader = csv.reader(StringIO(bytes.decode()))
    row = next(reader)
    return {data_id: list(map(float, row))}

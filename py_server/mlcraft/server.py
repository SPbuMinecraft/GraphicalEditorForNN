import requests
from http import HTTPStatus
from flask import Blueprint, request, current_app

from .utils import convert_model_parameters, is_valid_model, convert_model
from .check_dimensions import assert_dimensions_match

from .errors import Error

from .db import sql_worker
from .dataset import extract_predict_data, extract_train_data


app = Blueprint("make a better name", __name__)


@app.route("/user", methods=["POST"])
def add_user():
    json_data = request.json
    user_id = sql_worker.add_user(json_data)
    return {"user_id": user_id}, HTTPStatus.CREATED


@app.route("/user", methods=["PUT"])
def login_user():
    json_data = request.json
    user_id = sql_worker.get_user(json_data)
    return {"user_id": user_id}, HTTPStatus.OK


@app.route("/models/<int:user_id>")
def model_list(user_id: int):
    models = sql_worker.get_models_list(user_id)
    return models, HTTPStatus.OK


@app.route("/model/<int:user_id>", methods=["POST"])
def add_model(user_id: int):
    json_data = request.json
    inserted_id = sql_worker.add_model(user_id, json_data["name"])
    return {"model_id": inserted_id}, HTTPStatus.CREATED


@app.route("/<int:user_id>/<int:model_id>", methods=["GET", "PUT", "DELETE"])
def model(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    match request.method:
        case "GET":
            raw = sql_worker.get_raw_model(model_id)
            return raw, HTTPStatus.OK
        case "PUT":
            d: dict[str, str | None] = defaultdict(lambda: None, **request.json)  # type: ignore
            sql_worker.update_model(model_id, d["name"], d["raw"])
            return "", HTTPStatus.OK
        case "DELETE":
            sql_worker.delete_model(model_id)
            return "", HTTPStatus.OK


@app.route("/<int:user_id>/<int:model_id>/copy", methods=["PUT"])
def copy_model(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    id = sql_worker.copy_model(model_id)
    return {"model_id": id}, HTTPStatus.CREATED


@app.route("/<int:user_id>/<int:src_model_id>/copy/<int:dst_model_id>", methods=["PUT"])
def assign_model(user_id: int, src_model_id: int, dst_model_id: int):
    sql_worker.verify_access(user_id, src_model_id)
    sql_worker.verify_access(user_id, dst_model_id)
    sql_worker.copy_model(src_model_id, dst_model_id)
    return "", HTTPStatus.OK


@app.route("/layer/<int:user_id>/<int:model_id>", methods=["POST"])
def add_layer(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    json_data = request.json
    inserted_id = sql_worker.add_layer(
        json_data["type"], json_data["parameters"], model_id
    )
    return {"layer_id": inserted_id}, HTTPStatus.CREATED


@app.route("/layer/<int:user_id>/<int:model_id>", methods=["PUT"])
def update_layer(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    json_data = request.json
    sql_worker.update_layer(json_data["parameters"], int(json_data["id"]), model_id)
    return "", HTTPStatus.OK


@app.route("/clear_model/<int:user_id>/<int:model_id>", methods=["POST"])
def clear_model(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    sql_worker.clear_model(model_id)
    return "", HTTPStatus.OK


@app.route("/connection/<int:user_id>/<int:model_id>", methods=["POST"])
def add_connection(user_id: int, model_id: int):
    json_data = request.json
    layer_from_id, layer_to_id = int(json_data["layer_from"]), int(
        json_data["layer_to"]
    )
    sql_worker.verify_connection(user_id, model_id, layer_from_id, layer_to_id)
    sql_worker.add_connection(layer_from_id, layer_to_id, model_id)
    return "", HTTPStatus.CREATED


@app.route("/delete_layer/<int:user_id>/<int:model_id>", methods=["PUT"])
def delete_layer(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    json_data = request.json
    sql_worker.delete_layer(int(json_data["id"]), model_id)
    return "", HTTPStatus.OK


@app.route("/delete_connection/<int:user_id>/<int:model_id>", methods=["PUT"])
def delete_connection(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    json_data = request.json
    sql_worker.delete_connection(
        int(json_data["layer_from"]), int(json_data["layer_to"]), model_id
    )
    return "", HTTPStatus.OK


@app.route("/update_parents_order/<int:user_id>/<int:model_id>", methods=["PUT"])
def update_parents_order(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)
    json_data = request.json
    sql_worker.update_parents_order(
        json_data["new_parents"], int(json_data["layer_id"]), model_id
    )
    return "", HTTPStatus.OK


@app.route("/train/<int:user_id>/<int:model_id>/<int:safe>", methods=["PUT"])
def train_model(
    user_id: int, model_id: int, safe: int
):  # Unfortunately, flask don't have convertor for bool
    # checks belonging of the model to user
    sql_worker.verify_access(user_id, model_id)
    if not request.data:
        raise Error("No csv data provided")
    if sql_worker.is_model_trained(model_id) and safe:
        raise Error("Already trained", HTTPStatus.PRECONDITION_FAILED)

    model = sql_worker.get_graph_elements(model_id)
    convert_model_parameters(model)
    if not is_valid_model(model):
        raise Error("Invalid model found", HTTPStatus.NOT_ACCEPTABLE)
    assert_dimensions_match(model["layers"])

    dataset = extract_train_data(request.data, model)

    convert_model(model)
    model = {"graph": model, "dataset": dataset}

    response = requests.post(
        current_app.config["CPP_SERVER"] + f"/train/{model_id}",
        json=model,
        timeout=3,
    )
    sql_worker.train_model(model_id)

    return response.text, response.status_code


@app.route("/predict/<int:user_id>/<int:model_id>", methods=["PUT"])
def predict(user_id: int, model_id: int):
    sql_worker.verify_access(user_id, model_id)

    if not request.data:
        raise Error("No csv data provided")

    model = sql_worker.get_graph_elements(model_id)
    convert_model_parameters(model)
    json_data = extract_predict_data(request.data, model)

    if not sql_worker.is_model_trained(model_id):
        raise Error("Not trained", HTTPStatus.PRECONDITION_FAILED)

    response = requests.post(
        current_app.config["CPP_SERVER"] + f"/predict/{model_id}",
        json=json_data,
        timeout=3,
    )

    return response.text, response.status_code

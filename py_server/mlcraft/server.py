import requests
from http import HTTPStatus
from flask import Blueprint, request, current_app

from .utils import convert_model_parameters, is_valid_model, convert_model
from .check_dimensions import check_dimensions

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


@app.route("/model/<int:user_id>", methods=["POST"])
def add_model(user_id: int):
    json_data = request.json
    inserted_id = sql_worker.add_model(user_id, json_data["name"])
    return {"model_id": inserted_id}, HTTPStatus.CREATED


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
    check_dimensions(model["layers"])

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

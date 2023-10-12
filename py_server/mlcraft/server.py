import requests
from http import HTTPStatus
from flask import Blueprint, request, current_app

from .utils import (
    error, parse_parameters, is_valid_model,
    LayersConnectionStatus, DeleteStatus,
)
from .db import sql_worker


app = Blueprint("make a better name", __name__)


@app.route("/add_user", methods=["POST"])
def add_user():
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        inserted_id = sql_worker.add_user(json)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route("/add_model/<int:user_id>", methods=["POST"])
def add_model(user_id: int):
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        inserted_id = sql_worker.add_model(user_id, json["name"])
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route("/add_layer/<int:user_id>/<int:model_id>", methods=["POST"])
def add_layer(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")

    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        inserted_id = sql_worker.add_layer(json["type"], json["parameters"], model_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route("/add_connection/<int:user_id>/<int:model_id>", methods=["POST"])
def add_connection(user_id: int, model_id: int):
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        layer_from_id, layer_to_id = int(json["layer_from"]), int(json["layer_to"])
        allowed = sql_worker.verify_connection(
            user_id, model_id, layer_from_id, layer_to_id
        )
        if allowed == LayersConnectionStatus.DoNotExist:
            error(HTTPStatus.NOT_FOUND, "At least one of layers does not exist")
        if allowed == LayersConnectionStatus.AccessDenied:
            error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")
        if allowed == LayersConnectionStatus.DimensionsMismatch:
            error(HTTPStatus.PRECONDITION_FAILED, message="Dimensions do not match")
        inserted_id = sql_worker.add_connection(layer_from_id, layer_to_id, model_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route("/delete_layer/<int:user_id>/<int:model_id>", methods=["POST"])
def delete_layer(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        status = sql_worker.delete_layer(int(json["id"]), model_id)
        if status == DeleteStatus.ElementNotExist:
            error(HTTPStatus.NOT_FOUND, "Layer does not exist")
        if status == DeleteStatus.LayerNotFree:
            error(HTTPStatus.BAD_REQUEST, "The layer contains connections")
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return "done", HTTPStatus.OK


@app.route("/delete_connection/<int:user_id>/<int:model_id>", methods=["POST"])
def delete_connection(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        status = sql_worker.delete_connection(int(json["id"]), model_id)
        if status == DeleteStatus.ElementNotExist:
            error(HTTPStatus.NOT_FOUND, "Connection does not exist")
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return "done", HTTPStatus.OK


@app.route("/train/<int:user_id>/<int:model_id>", methods=["POST"])
def train_model(user_id: int, model_id: int):
    # checks belonging of the model to user
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for training this model")

    try:
        model = sql_worker.get_graph_elements(model_id)
        for i in range(len(model["layers"])):
            model["layers"][i]["parameters"] = parse_parameters(
                model["layers"][i]["parameters"]
            )
        if not is_valid_model(model):
            error(HTTPStatus.NOT_ACCEPTABLE, "Invalid model found")
        # response = requests.Response(text="OK", status_code=HTTPStatus.OK)
        # requests.post(CPP_SERVER_ADDRESS + "/train", json=jsonify(model), timeout=3)  # Timeout=?
        # What does 'train' method (C++) return?
        return model  # return "Training finished", HTTPStatus.OK
    except TimeoutError as e:
        error(HTTPStatus.REQUEST_TIMEOUT, "Training time limit exceeded")
    # Observe possible errors and catch them
    # except Exception as e:
    #     error(HTTPStatus.BAD_REQUEST, message=str(e))


# dummy: TODO (in sprint 2)
# Hardcoded function to make predictions for XOR model
@app.route("/predict/<int:user_id>/<int:model_id>", methods=["POST"])
def predict(user_id: int, model_id: int):
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        response = requests.post(
            current_app.config["CPP_SERVER"] + "/predict",
            json={"x": json["x"], "y": json["y"]},
            timeout=3,
        )
        return response.text, HTTPStatus.OK
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, str(e))
    except TimeoutError as e:
        error(HTTPStatus.REQUEST_TIMEOUT, "Time limit exceeded")
    except Exception as e:
        error(HTTPStatus.BAD_REQUEST, str(e))

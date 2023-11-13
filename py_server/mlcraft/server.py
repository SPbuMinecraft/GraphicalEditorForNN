from json import dumps
from sqlite3 import IntegrityError
import requests
from http import HTTPStatus
from flask import Blueprint, request, current_app

from .utils import (
    error,
    parse_parameters,
    is_valid_model,
    LayersConnectionStatus,
    DeleteStatus,
)
from . import errors

from .db import sql_worker
from .dataset import extract_predict_data, extract_train_data


app = Blueprint("make a better name", __name__)


@app.route("/add_user", methods=["POST"])
def add_user():
    json_data = request.get_json()
    if not json_data:
        return {"error": "No JSON data provided"}, HTTPStatus.BAD_REQUEST
    try:
        user_id = sql_worker.add_user(json_data)
        return {"user_id": user_id}, HTTPStatus.CREATED
    except errors.UserAlreadyExistsError as e:
        return (
            {"error": str(e), "problemPart": "username"},
            HTTPStatus.CONFLICT,
        )  # HTTP 409 Conflict\
    except errors.MailAlreadyExistsError as e:
        return {"error": str(e), "problemPart": "mail"}, HTTPStatus.CONFLICT
    except IntegrityError as e:
        return (
            {"error": str(e)},
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )  # HTTP 500 Internal Server Error


@app.route("/login_user", methods=["POST"])
def login_user():
    json_data = request.get_json()
    if not json_data:
        return {"error": "No JSON data provided"}, HTTPStatus.BAD_REQUEST
    try:
        user_id = sql_worker.get_user(json_data)
        return {"user_id": user_id}, HTTPStatus.OK
    except errors.UserNotFoundError as e:
        return (
            {"error": str(e), "problemPart": "username"},
            HTTPStatus.UNAUTHORIZED,
        )
    except errors.WrongPasswordError as e:
        return (
            {"error": str(e), "problemPart": "password"},
            HTTPStatus.UNAUTHORIZED,
        )


@app.route("/add_model/<int:user_id>", methods=["POST"])
def add_model(user_id: int):
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        inserted_id = sql_worker.add_model(user_id, json["name"])
        if inserted_id == -1:
            error(HTTPStatus.BAD_REQUEST, message="No user with that id")
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


@app.route("/update_layer/<int:user_id>/<int:model_id>", methods=["PUT"])
def update_layer(user_id: int, model_id: int):
    if not sql_worker.verify_access(user_id, model_id):
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")
    json = request.json
    if json is None:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        sql_worker.update_layer(json["parameters"], int(json["id"]), model_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, f"Key error: {e}")
    except StopIteration as e:
        error(HTTPStatus.BAD_REQUEST, f"No such id: {json['id']}")
    return "done", HTTPStatus.OK


@app.route("/clear_model/<int:user_id>/<int:model_id>", methods=["POST"])
def clear_model(user_id: int, model_id: int):
    if not sql_worker.verify_access(user_id, model_id):
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")
    try:
        sql_worker.clear_model(model_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return "done", HTTPStatus.OK


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
        if allowed == LayersConnectionStatus.WrongDirection:
            error(
                HTTPStatus.PRECONDITION_FAILED,
                message="Wrong direction in data or output layer",
            )
        if allowed == LayersConnectionStatus.Cycle:
            error(HTTPStatus.BAD_REQUEST, "Graph must by acyclic")
        status = sql_worker.add_connection(layer_from_id, layer_to_id, model_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(status), HTTPStatus.CREATED


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
        status = sql_worker.delete_connection(
            int(json["layer_from"]), int(json["layer_to"]), model_id
        )
        if status == DeleteStatus.ElementNotExist:
            error(HTTPStatus.NOT_FOUND, "Connection does not exist")
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return "done", HTTPStatus.OK


@app.route("/update_parents_order/<int:user_id>/<int:model_id>", methods=["POST"])
def update_parents_order(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        status = sql_worker.update_parents_order(
            json["new_parents"], int(json["layer_id"]), model_id
        )
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return "done", HTTPStatus.OK


@app.route("/train/<int:user_id>/<int:model_id>/<int:safe>", methods=["POST"])
def train_model(
    user_id: int, model_id: int, safe: int
):  # Unfortunately, flask don't have convertor for bool
    # checks belonging of the model to user
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for training this model")
    if not request.data:
        error(HTTPStatus.BAD_REQUEST, "No csv file provided")
    dataset = extract_train_data(request.data, model_id)
    try:
        if sql_worker.is_model_trained(model_id) and safe:
            error(HTTPStatus.PRECONDITION_FAILED, "Already trained")
        model = sql_worker.get_graph_elements(model_id)
        for i in range(len(model["layers"])):
            model["layers"][i]["parameters"] = parse_parameters(
                model["layers"][i]["parameters"]
            )
        if not is_valid_model(model):
            error(HTTPStatus.NOT_ACCEPTABLE, "Invalid model found")
        # Convert json to another format for C++ by deleting connsetcions ids and rename layers_type
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

        model = {"graph": model, "dataset": dataset}
        response = requests.post(
            current_app.config["CPP_SERVER"] + f"/train/{model_id}",
            json=model,
            timeout=3,
        )
        sql_worker.train_model(model_id)
        return response.text, response.status_code
    except requests.exceptions.ConnectionError as e:
        error(HTTPStatus.INTERNAL_SERVER_ERROR, "No c++ server found")
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, str(e))
    except TimeoutError as e:
        error(HTTPStatus.REQUEST_TIMEOUT, "Training time limit exceeded")


@app.route("/predict/<int:user_id>/<int:model_id>", methods=["POST"])
def predict(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for training this model")
    if not request.data:
        error(HTTPStatus.BAD_REQUEST, message="No csv file provided")
    json = extract_predict_data(request.data, model_id)
    try:
        if not sql_worker.is_model_trained(model_id):
            error(HTTPStatus.PRECONDITION_FAILED, "Not trained")
        response = requests.post(
            current_app.config["CPP_SERVER"] + f"/predict/{model_id}",
            json=json,
            # timeout=3,
        )
        return response.text, response.status_code
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, str(e))
    except TimeoutError as e:
        error(HTTPStatus.REQUEST_TIMEOUT, "Time limit exceeded")

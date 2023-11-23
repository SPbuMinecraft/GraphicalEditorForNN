from json import dumps
from sqlite3 import IntegrityError
import requests
from http import HTTPStatus
from flask import Blueprint, request, current_app

from .utils import (
    error,
    parse_parameters,
    is_valid_model,
    check_dimensions,
    LayersConnectionStatus,
    DeleteStatus,
    VerificationStatus,
    DimensionsCheckStatus,
)
from . import errors

from .db import sql_worker
from .dataset import extract_predict_data, extract_train_data


app = Blueprint("make a better name", __name__)


@app.route("/user", methods=["POST"])
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
        )  # HTTP 409 Conflict
    except errors.MailAlreadyExistsError as e:
        return {"error": str(e), "problemPart": "mail"}, HTTPStatus.CONFLICT
    except IntegrityError as e:
        return (
            {"error": str(e)},
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )  # HTTP 500 Internal Server Error
    except KeyError as e:
        return {"error": f"Missing required field {str(e)}"}, HTTPStatus.BAD_REQUEST


@app.route("/user", methods=["PUT"])
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
    except KeyError as e:
        return {"error": f"Missing required field {str(e)}"}, HTTPStatus.BAD_REQUEST


@app.route("/model/<int:user_id>", methods=["POST"])
def add_model(user_id: int):
    json = request.json
    if not json:
        return {"error": "No JSON data provided"}, HTTPStatus.BAD_REQUEST
    try:
        inserted_id = sql_worker.add_model(user_id, json["name"])
        return {"model_id": inserted_id}, HTTPStatus.CREATED
    except errors.ObjectNotFoundError as e:
        return {"error": str(e)}, HTTPStatus.NOT_FOUND
    except KeyError as e:
        return {"error": f"Missing required field {str(e)}"}, HTTPStatus.BAD_REQUEST


@app.route("/layer/<int:user_id>/<int:model_id>", methods=["POST"])
def add_layer(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for changing this model"
        }, HTTPStatus.FORBIDDEN
    json = request.json
    if not json:
        return {"error": "No JSON data provided"}, HTTPStatus.BAD_REQUEST
    try:
        inserted_id = sql_worker.add_layer(json["type"], json["parameters"], model_id)
        return {"layer_id": inserted_id}, HTTPStatus.CREATED
    except KeyError as e:
        return {"error": f"Missing required field {str(e)}"}, HTTPStatus.BAD_REQUEST


@app.route("/layer/<int:user_id>/<int:model_id>", methods=["PUT"])
def update_layer(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for changing this model"
        }, HTTPStatus.FORBIDDEN
    json = request.json
    if json is None:
        return {"error": "No JSON data provided"}, HTTPStatus.BAD_REQUEST
    try:
        sql_worker.update_layer(json["parameters"], int(json["id"]), model_id)
        return "", HTTPStatus.OK
    except KeyError as e:
        return {"error": f"Key error: {e}"}, HTTPStatus.BAD_REQUEST
    except StopIteration as e:
        return {"error": f"No layer with such id: {json['id']}"}, HTTPStatus.NOT_FOUND


@app.route("/clear_model/<int:user_id>/<int:model_id>", methods=["POST"])
def clear_model(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for changing this model"
        }, HTTPStatus.FORBIDDEN
    sql_worker.clear_model(model_id)
    return "", HTTPStatus.OK


@app.route("/connection/<int:user_id>/<int:model_id>", methods=["POST"])
def add_connection(user_id: int, model_id: int):
    json = request.json
    if not json:
        {"error": "No Json data provided"}, HTTPStatus.BAD_REQUEST
    try:
        layer_from_id, layer_to_id = int(json["layer_from"]), int(json["layer_to"])
        allowed = sql_worker.verify_connection(
            user_id, model_id, layer_from_id, layer_to_id
        )
        if allowed == LayersConnectionStatus.DoNotExist:
            return {
                "error": "At least one of layers does not exist"
            }, HTTPStatus.NOT_FOUND
        if allowed == LayersConnectionStatus.AccessDenied:
            return {
                "error": "You have no rights for changing this model"
            }, HTTPStatus.FORBIDDEN
        if allowed == LayersConnectionStatus.WrongDirection:
            return {
                "error": "Wrong direction in data or output layer"
            }, HTTPStatus.PRECONDITION_FAILED
        if allowed == LayersConnectionStatus.Cycle:
            return {"error": "Graph must by acyclic"}, HTTPStatus.PRECONDITION_FAILED
        status = sql_worker.add_connection(layer_from_id, layer_to_id, model_id)
        if status == -1:
            return {"error": "Model not found"}, HTTPStatus.NOT_FOUND
        return "", HTTPStatus.CREATED
    except KeyError as e:
        return {"error": str(e)}, HTTPStatus.BAD_REQUEST


@app.route("/delete_layer/<int:user_id>/<int:model_id>", methods=["PUT"])
def delete_layer(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for changing this model"
        }, HTTPStatus.FORBIDDEN
    json = request.json
    if not json:
        return {"error": "No Json data provided"}, HTTPStatus.BAD_REQUEST
    try:
        status = sql_worker.delete_layer(int(json["id"]), model_id)
        if status == DeleteStatus.ElementNotExist:
            return {"error": "Layer does not exist"}, HTTPStatus.NOT_FOUND
        if status == DeleteStatus.LayerNotFree:
            {"error": "The layer contains connections"}, HTTPStatus.PRECONDITION_FAILED
        return "", HTTPStatus.OK
    except KeyError as e:
        return {"error": str(e)}, HTTPStatus.BAD_REQUEST


@app.route("/delete_connection/<int:user_id>/<int:model_id>", methods=["PUT"])
def delete_connection(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for changing this model"
        }, HTTPStatus.FORBIDDEN
    json = request.json
    if not json:
        return {"error": "No Json data provided"}, HTTPStatus.BAD_REQUEST
    try:
        status = sql_worker.delete_connection(
            int(json["layer_from"]), int(json["layer_to"]), model_id
        )
        if status == DeleteStatus.ElementNotExist:
            return {"error": "Connection does not exist"}, HTTPStatus.NOT_FOUND
        return "", HTTPStatus.OK
    except KeyError as e:
        return {"error": str(e)}, HTTPStatus.BAD_REQUEST


@app.route("/update_parents_order/<int:user_id>/<int:model_id>", methods=["PUT"])
def update_parents_order(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for changing this model"
        }, HTTPStatus.FORBIDDEN
    json = request.json
    if not json:
        return {"error": "No Json data provided"}, HTTPStatus.BAD_REQUEST
    try:
        status = sql_worker.update_parents_order(
            json["new_parents"], int(json["layer_id"]), model_id
        )
        return "", HTTPStatus.OK
    except KeyError as e:
        return {"error": str(e)}, HTTPStatus.BAD_REQUEST
    except errors.ObjectNotFoundError as e:
        return {"error": str(e)}, HTTPStatus.NOT_FOUND


@app.route("/train/<int:user_id>/<int:model_id>/<int:safe>", methods=["PUT"])
def train_model(
    user_id: int, model_id: int, safe: int
):  # Unfortunately, flask don't have convertor for bool
    # checks belonging of the model to user
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for training this model"
        }, HTTPStatus.FORBIDDEN
    if not request.data:
        return {"error": "No csv data provided"}, HTTPStatus.BAD_REQUEST
    dataset = extract_train_data(request.data, model_id)
    try:
        if sql_worker.is_model_trained(model_id) and safe:
            return {"error": "Already trained"}, HTTPStatus.PRECONDITION_FAILED
        model = sql_worker.get_graph_elements(model_id)
        for i in range(len(model["layers"])):
            model["layers"][i]["parameters"] = parse_parameters(
                model["layers"][i]["parameters"]
            )
        if not is_valid_model(model):
            return {"error": "Invalid model found"}, HTTPStatus.NOT_ACCEPTABLE
        # Convert json to another format for C++ by deleting connsetcions ids and rename layers_type
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
        dimensions_status, layer_id = check_dimensions(model["layers"])
        if dimensions_status == DimensionsCheckStatus.InvalidNumberOfInputs:
            return {"error": f"Invalid number of inputs for layer {layer_id}"}, HTTPStatus.NOT_ACCEPTABLE
        elif dimensions_status == DimensionsCheckStatus.DimensionsMismatch:
            return {"error": f"Layer {layer_id} does not match with it's parents in dimensions"}, \
                                                                                HTTPStatus.NOT_ACCEPTABLE
        model["connections"] = list(
            {
                "layer_from": layer_from,
                "layer_to": layer_to["id"],
            }
            for layer_to in model["layers"]
            for layer_from in layer_to["parents"]
        )  # Create list of connections for C++ server

        model = {"graph": model, "dataset": dataset}
        response = requests.post(
            current_app.config["CPP_SERVER"] + f"/train/{model_id}",
            json=model,
            # timeout=3,
        )
        sql_worker.train_model(model_id)
        return response.text, response.status_code
    except requests.exceptions.ConnectionError as e:
        return {"error": "No c++ server found"}, HTTPStatus.INTERNAL_SERVER_ERROR
    except KeyError as e:
        return {"error": str(e)}, HTTPStatus.BAD_REQUEST
    except TimeoutError as e:
        return {"error": "Training time limit exceeded"}, HTTPStatus.REQUEST_TIMEOUT


@app.route("/predict/<int:user_id>/<int:model_id>", methods=["PUT"])
def predict(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if allowed == VerificationStatus.NotFound:
        return {"error": "Model does not exist"}, HTTPStatus.NOT_FOUND
    if allowed == VerificationStatus.Forbidden:
        return {
            "error": "You have no rights for using this model"
        }, HTTPStatus.FORBIDDEN
    if not request.data:
        return {"error": "No csv data provided"}, HTTPStatus.BAD_REQUEST
    json = extract_predict_data(request.data, model_id)
    try:
        if not sql_worker.is_model_trained(model_id):
            return {"error": "Not trained"}, HTTPStatus.PRECONDITION_FAILED
        response = requests.post(
            current_app.config["CPP_SERVER"] + f"/predict/{model_id}",
            json=json,
            # timeout=3,
        )
        return response.text, response.status_code
    except requests.exceptions.ConnectionError as e:
        return {"error": "No c++ server found"}, HTTPStatus.INTERNAL_SERVER_ERROR
    except KeyError as e:
        return {"error": str(e)}, HTTPStatus.BAD_REQUEST
    except TimeoutError as e:
        return {"error": "Time limit exceeded"}, HTTPStatus.REQUEST_TIMEOUT

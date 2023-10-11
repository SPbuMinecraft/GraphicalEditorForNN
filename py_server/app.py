import requests
from datetime import datetime
from http import HTTPStatus
from flask import Flask, abort, request, Response, jsonify

import os
import sys
import json
import inspect

# Maybe there is a normal solution, who knows??
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import error, parse_parameters, is_valid_model, LayersConnectionStatus
from db.app import sql_worker

app = Flask(__name__)
app.config.from_pyfile("../config.py")

DB_ADDRESS = "http://" + app.config["DB_HOSTNAME"] + ":" + str(app.config["DB_PORT"])
CPP_SERVER_ADDRESS = "http://" + app.config["SERVER_HOSTNAME"] + ":" + str(app.config["SERVER_PORT"])


def get_graph_elements(model_id: int):
    layers = [{
        'type': layer.layer_type,
        'parameters': layer.parameters,
    } for layer in sql_worker.get_model_layers(model_id)]
    connections = [{
        'layer_from': connection.layer_from,
        'layer_to': connection.layer_to
    } for connection in sql_worker.get_model_connections(model_id)]
    return {'layers': layers, 'connections': connections}


@app.route('/add_user', methods=['POST'])
def add_user():
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        inserted_id = sql_worker.add_user(json)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route('/add_model/<int:user_id>', methods=['POST'])
def add_model(user_id: int):
    try:
        inserted_id = sql_worker.add_model(user_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route('/add_layer/<int:user_id>/<int:model_id>', methods=['POST'])
def add_layer(user_id: int, model_id: int):
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for changing this model")

    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        inserted_id = sql_worker.add_layer(json['type'], json['parameters'], model_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route('/add_connection/<int:user_id>', methods=['POST'])
def add_connection(user_id: int):
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        layer_from_id, layer_to_id = int(json['layer_from']), int(json['layer_to'])
        allowed = sql_worker.verify_connection(user_id, layer_from_id, layer_to_id)
        if allowed == LayersConnectionStatus.DoNotExist:
            error(HTTPStatus.NOT_FOUND, "At leat on of layers does not exist")
        if allowed == LayersConnectionStatus.FromDifferentModels:
            error(HTTPStatus.PRECONDITION_FAILED, "Layers refer to different models")
        if allowed == LayersConnectionStatus.AccessDenied:
            error(HTTPStatus.FORBIDDEN, "You have no rights for training this model")
        if allowed == LayersConnectionStatus.DimensionsMismatch:
            error(HTTPStatus.PRECONDITION_FAILED, message="Dimensions do not match")
        inserted_id = sql_worker.add_connection(layer_from_id, layer_to_id)
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    return str(inserted_id), HTTPStatus.CREATED


@app.route('/train/<int:user_id>/<int:model_id>', methods=['POST'])
def train_model(user_id: int, model_id: int):
    # checks belonging of the model to user
    allowed = sql_worker.verify_access(user_id, model_id)
    if not allowed:
        error(HTTPStatus.FORBIDDEN, "You have no rights for training this model")

    try:
        model = get_graph_elements(model_id)
        for i in range(len(model['layers'])):
            model['layers'][i]['parameters'] = parse_parameters(model['layers'][i]['parameters'])
        if not is_valid_model(model):
            error(HTTPStatus.NOT_ACCEPTABLE, "Invalid model found")
        # response = requests.Response(text="OK", status_code=HTTPStatus.OK)
        # requests.post(CPP_SERVER_ADDRESS + "/train", json=jsonify(model), timeout=3)  # Timeout=?
        # What does 'train' method (C++) return?
        return jsonify(model)  # return "Training finished", HTTPStatus.OK
    except TimeoutError as e:
        error(HTTPStatus.REQUEST_TIMEOUT, "Training time limit exceeded")
    # Observe possible errors and catch them
    # except Exception as e:
    #     error(HTTPStatus.BAD_REQUEST, message=str(e))


# dummy: TODO (in sprint 2)
# Hardcoded function to make predictions for XOR model
@app.route('/predict/<int:user_id>/<int:model_id>', methods=['POST'])
def predict(user_id: int, model_id: int):
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        response = requests.post(
            CPP_SERVER_ADDRESS + "/predict",
            json=jsonify({"x": json["x"], "y": json["y"]}).json,
            timeout=3
        )
        return response.text, HTTPStatus.OK
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, str(e))
    except TimeoutError as e:
        error(HTTPStatus.REQUEST_TIMEOUT, "Time limit exceeded")
    except Exception as e:
        error(HTTPStatus.BAD_REQUEST, str(e))


if __name__ == "__main__":
    app.run(host="localhost", port=app.config["PY_SERVER_PORT"], debug=True)

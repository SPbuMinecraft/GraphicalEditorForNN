import json
from sqlite3 import IntegrityError
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from .utils import (
    LayersConnectionStatus,
    DeleteStatus,
    parse_parameters,
    check_paths_exist,
)
from . import errors

db = SQLAlchemy()  # Has to be global by Flask documentation


class User(db.Model):  # type: ignore
    __tablename__ = "users_table"

    id = db.Column(db.Integer, primary_key=True)
    login = db.Column(db.Text)
    password = db.Column(db.Text)
    mail = db.Column(db.Text)
    model = db.relationship("Model", backref="user_id")


class Model(db.Model):  # type: ignore
    __tablename__ = "models_table"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text)
    owner = db.Column(db.Integer, db.ForeignKey("users_table.id"), nullable=False)
    content = db.Column(db.Text)
    is_trained = db.Column(db.Boolean)


class SQLWorker:
    def __init__(self, app):
        with app.app_context():
            db.create_all()

    def add_user(self, user_parameters: dict):
        with current_app.app_context():
            # Не очень хорошее решение по обработке ошибок... А если ошибок станет 10, 100?
            existing_user = User.query.filter_by(login=user_parameters["login"]).first()
            if existing_user is not None:
                raise errors.UserAlreadyExistsError(
                    "There's already an account with this username"
                )

            existing_mail = User.query.filter_by(mail=user_parameters["mail"]).first()
            if existing_mail is not None:
                raise errors.MailAlreadyExistsError(
                    "There's already an account with this email."
                )

            new_user = User()
            new_user.login = user_parameters["login"]
            new_user.password = user_parameters["password"]
            new_user.mail = user_parameters["mail"]

            try:
                db.session.add(new_user)
                db.session.commit()
                return new_user.id
            except IntegrityError as e:
                db.session.rollback()
                raise IntegrityError(
                    "Failed to add the user due to a uniqueness violation"
                )

    def get_user(self, user_parameters: dict):
        existing_user = User.query.filter_by(login=user_parameters["login"]).first()
        if existing_user is None:
            raise errors.UserNotFoundError("User not found")
        if existing_user.password != user_parameters["password"]:
            raise errors.WrongPasswordError("Wrong password")
        return existing_user.id

    def add_model(self, user_id: int, name: str):
        with current_app.app_context():
            model_owner = db.session.get(User, user_id)
            if not model_owner:
                return -1
            new_model = Model(
                owner=user_id,
                name=name,
                content='{"layers": [], "connections": []}',
                is_trained=False,
            )
            db.session.add(new_model)
            db.session.commit()
            return new_model.id

    def add_layer(self, layer_type: str, parameters: str, model_id: int):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return -1
            model_items = json.loads(
                model.content
            )  # Throws exception if model.content is not json
            new_id = (
                model_items["layers"][-1]["id"] + 1
                if len(model_items["layers"]) > 0
                else 0
            )
            new_layer = {
                "id": new_id,
                "layer_type": layer_type,
                "parameters": parameters,
            }  # Should be refactored?
            model_items["layers"].append(new_layer)
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()
            return new_id

    def update_layer(self, new_params: str, layer_id: int, model_id: int):
        with current_app.app_context():
            model = db.session.get(Model, model_id)
        if model is None:
            # change later for suitable Error type
            raise KeyError(f"No model with id {model_id}")
        items = json.loads(model.content)
        layer = next(l for l in items["layers"] if l["id"] == layer_id)
        layer["parameters"] = new_params
        model.content = json.dumps(items)
        with current_app.app_context():
            db.session.add(model)
            db.session.commit()

    def add_connection(
        self, layer_from: int, layer_to: int, model_id: int
    ):  # When we sure, that adding is correct
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return -1
            model_items = json.loads(model.content)
            new_id = (
                model_items["connections"][-1]["id"] + 1
                if len(model_items["connections"]) > 0
                else 0
            )
            new_connection = {
                "id": new_id,
                "layer_from": layer_from,
                "layer_to": layer_to,
            }
            model_items["connections"].append(new_connection)
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()
            return new_id

    def delete_layer(self, layer_id: int, model_id: int):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return DeleteStatus.ModelNotExist
            model_items = json.loads(model.content)
            if any(
                conn["layer_from"] == layer_id or conn["layer_to"] == layer_id
                for conn in model_items["connections"]
            ):
                return DeleteStatus.LayerNotFree

            new_layers_list = list(
                filter(lambda layer: layer["id"] != layer_id, model_items["layers"])
            )
            if len(new_layers_list) == len(model_items["layers"]):
                return DeleteStatus.ElementNotExist
            model_items["layers"] = new_layers_list
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()
            return DeleteStatus.OK

    def delete_connection(self, connection_id: int, model_id: int):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return DeleteStatus.ModelNotExist
            model_items = json.loads(model.content)
            new_connections_list = list(
                filter(
                    lambda connection: connection["id"] != connection_id,
                    model_items["connections"],
                )
            )
            if len(new_connections_list) == len(model_items["connections"]):
                return DeleteStatus.ElementNotExist
            model_items["connections"] = new_connections_list
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()
            return DeleteStatus.OK

    def clear_model(self, model_id: int):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return DeleteStatus.ModelNotExist
            model_items = json.loads(model.content)
            model_items["connections"] = []
            model_items["layers"] = []
            model.content = json.dumps(model_items)
            db.session.add(model)
            db.session.commit()
            return DeleteStatus.OK

    def check_dimensions(self, layer_from: dict, layer_to: dict):
        parameters_from = parse_parameters(layer_from["parameters"])
        parameters_to = parse_parameters(layer_to["parameters"])
        return int(parameters_from["outputs"]) == int(parameters_to["inputs"])

    def verify_connection(
        self, user_id: int, model_id: int, layer_from: int, layer_to: int
    ):
        with current_app.app_context():
            if not self.verify_access(user_id, model_id):
                return LayersConnectionStatus.AccessDenied
            model = Model.query.get(model_id)
            model_items = json.loads(model.content)
            layers = model_items["layers"]
            layer1_candidates = list(
                filter(lambda layer: layer["id"] == layer_from, layers)
            )
            layer2_candidates = list(
                filter(lambda layer: layer["id"] == layer_to, layers)
            )
            if not layer1_candidates or not layer2_candidates:
                return LayersConnectionStatus.DoNotExist
            layer1 = layer1_candidates[0]
            layer2 = layer2_candidates[0]
            if not self.check_dimensions(layer1, layer2):
                return LayersConnectionStatus.DimensionsMismatch
            if layer2["layer_type"] == "Data" or layer1["layer_type"] == "Output":
                return LayersConnectionStatus.WrongDirection
            if check_paths_exist([layer_to], [layer_from], model_items):
                return LayersConnectionStatus.Cycle
            return LayersConnectionStatus.OK

    def verify_access(self, user_id, model_id):
        with current_app.app_context():
            model_passport = Model.query.get(model_id)
            if not model_passport or model_passport.owner != user_id:
                return False
            return True

    def get_graph_elements(self, model_id):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return -1
            model_items = json.loads(model.content)
            return model_items

    def train_model(self, model_id: int):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return -1
            model.is_trained = True
            db.session.add(model)
            db.session.commit()
            return 1

    def is_model_trained(self, model_id: int):
        with current_app.app_context():
            model = Model.query.get(model_id)
            if not model:
                return False
            return model.is_trained


sql_worker: SQLWorker = None  # type: ignore


def init_app(app):
    global sql_worker
    db.init_app(app)
    sql_worker = SQLWorker(app)  # Has to be there

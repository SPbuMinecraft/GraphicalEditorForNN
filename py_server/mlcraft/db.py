from threading import Lock  # fuck it, just using old robust methods
import json
from sqlite3 import IntegrityError
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from .utils import parse_parameters, check_paths_exist
from .errors import Error
from http import HTTPStatus

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
    """
    Important: whenever you work with 'content' column of the model,
    you must first acquire content_lock, release it after you commit the session
    Tip: with mutex: ...  # also works
    """

    def __init__(self, app):
        self.content_lock = Lock()
        with app.app_context():
            db.create_all()

    def add_user(self, user_parameters: dict):
        with current_app.app_context():
            # Не очень хорошее решение по обработке ошибок... А если ошибок станет 10, 100?
            existing_user = User.query.filter_by(login=user_parameters["login"]).first()
            if existing_user is not None:
                raise Error(
                    "There's already an account with this username",
                    HTTPStatus.CONFLICT,
                    problemPart="username",
                )

            existing_mail = User.query.filter_by(mail=user_parameters["mail"]).first()
            if existing_mail is not None:
                raise Error(
                    "There's already an account with this email.",
                    HTTPStatus.CONFLICT,
                    problemPart="mail",
                )

            new_user = User()
            new_user.login = user_parameters["login"]
            new_user.password = user_parameters["password"]
            new_user.mail = user_parameters["mail"]

            try:
                db.session.add(new_user)
                db.session.commit()
                return new_user.id
            except IntegrityError:
                # TODO: how is this gonna happen?
                db.session.rollback()
                raise Error(
                    "Failed to add the user due to a uniqueness violation",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    def get_user(self, user_parameters: dict):
        existing_user = User.query.filter_by(login=user_parameters["login"]).first()
        if existing_user is None:
            raise Error(
                "User not found", HTTPStatus.UNAUTHORIZED, problemPart="username"
            )
        if existing_user.password != user_parameters["password"]:
            raise Error(
                "User not found", HTTPStatus.UNAUTHORIZED, problemPart="password"
            )
        return existing_user.id

    def add_model(self, user_id: int, name: str):
        with current_app.app_context():
            model_owner = db.session.get(User, user_id)
            if model_owner is None:
                raise Error(f"No user with id {user_id}", HTTPStatus.NOT_FOUND)
            new_model = Model(
                owner=user_id,
                name=name,
                content='{"layers": []}',
                is_trained=False,
            )
            db.session.add(new_model)
            db.session.commit()
            return new_model.id

    def add_layer(self, layer_type: str, parameters: str, model_id: int):
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
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
                "type": layer_type,
                "parameters": parameters,
                "parents": [],
            }  # Should be refactored?
            model_items["layers"].append(new_layer)
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()
            return new_id

    def update_layer(self, new_params: str, layer_id: int, model_id: int):
        with self.content_lock:
            model = self.get_model(model_id)
            items = json.loads(model.content)
            layer = next((l for l in items["layers"] if l["id"] == layer_id), None)
            if layer is None:
                raise Error(f"No layer with such id: {layer_id}", HTTPStatus.NOT_FOUND)
            layer["parameters"] = new_params
            model.content = json.dumps(items)
            model.is_trained = False
            with current_app.app_context():
                db.session.add(model)
                db.session.commit()

    def update_parents_order(self, new_parents: list, layer_id: int, model_id: int):
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            layer = next(l for l in model_items["layers"] if l["id"] == layer_id)
            if sorted(layer["parents"]) != sorted(new_parents):
                raise Error("Old connections does not match with new ones")
            layer["parents"] = new_parents
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()

    def add_connection(
        self, layer_from: int, layer_to: int, model_id: int
    ):  # When we sure, that adding is correct
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            layer = next(l for l in model_items["layers"] if l["id"] == layer_to)
            layer["parents"].append(layer_from)
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()

    def delete_layer(self, layer_id: int, model_id: int):
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            layer = next(
                iter(l for l in model_items["layers"] if l["id"] == layer_id), None
            )
            if layer is None:
                raise Error("Layer does not exist", HTTPStatus.NOT_FOUND)
            model_items["layers"].remove(layer)
            for layer in model_items["layers"]:
                layer["parents"] = list(
                    filter(lambda layer_from: layer_id != layer_from, layer["parents"])
                )
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()

    def delete_connection(self, layer_from: int, layer_to: int, model_id: int):
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            layer = next(
                iter(l for l in model_items["layers"] if l["id"] == layer_to), None
            )
            if layer is None or layer_from not in layer["parents"]:
                raise Error("Connection does not exist", HTTPStatus.NOT_FOUND)
            layer["parents"].remove(layer_from)
            model.content = json.dumps(model_items)
            model.is_trained = False
            db.session.add(model)
            db.session.commit()

    def clear_model(self, model_id: int):
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            model_items["layers"] = []
            model.content = json.dumps(model_items)
            db.session.add(model)
            db.session.commit()

    def get_model(self, model_id: int) -> Model:
        """Assuming self.content_lock is held"""
        with current_app.app_context():
            model = db.session.get(Model, model_id)
            if model is None:
                raise Error(f"No model with id {model_id}", HTTPStatus.NOT_FOUND)
            return model

    def check_dimensions(self, layer_from: dict, layer_to: dict):
        parameters_from = parse_parameters(layer_from["parameters"])
        parameters_to = parse_parameters(layer_to["parameters"])
        # return int(parameters_from["outputs"]) <= int(parameters_to["inputs"])
        return True

    def verify_connection(
        self, user_id: int, model_id: int, layer_from: int, layer_to: int
    ):
        with current_app.app_context(), self.content_lock:
            self.verify_access(user_id, model_id)
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            layers = model_items["layers"]
            layer1_candidates = list(
                filter(lambda layer: layer["id"] == layer_from, layers)
            )
            layer2_candidates = list(
                filter(lambda layer: layer["id"] == layer_to, layers)
            )
            if not layer1_candidates or not layer2_candidates:
                raise Error(
                    "At least one of layers does not exist", HTTPStatus.NOT_FOUND
                )
            layer1 = layer1_candidates[0]
            layer2 = layer2_candidates[0]
            # TO BE DONE later
            # if not self.check_dimensions(layer1, layer2):
            #     return LayersConnectionStatus.DimensionsMismatch
            if layer2["type"] == "Data" or layer1["type"] == "Output":
                raise Error(
                    "Wrong direction in data or output layer",
                    HTTPStatus.PRECONDITION_FAILED,
                )
            if check_paths_exist([layer_to], [layer_from], model_items):
                raise Error("Graph must by acyclic", HTTPStatus.PRECONDITION_FAILED)

    def verify_access(self, user_id, model_id):
        with current_app.app_context():
            model_passport = db.session.get(Model, model_id)
            if model_passport is None:
                raise Error("Model does not exist", HTTPStatus.NOT_FOUND)
            if model_passport.owner != user_id:
                raise Error(
                    "You have no rights for changing this model", HTTPStatus.FORBIDDEN
                )

    def get_graph_elements(self, model_id):
        with current_app.app_context(), self.content_lock:
            model = self.get_model(model_id)
            model_items = json.loads(model.content)
            return model_items

    def train_model(self, model_id: int):
        with current_app.app_context():
            model = self.get_model(model_id)
            model.is_trained = True
            db.session.add(model)
            db.session.commit()

    def is_model_trained(self, model_id: int):
        with current_app.app_context():
            model = self.get_model(model_id)
            return model.is_trained


sql_worker: SQLWorker = None  # type: ignore


def init_app(app):
    global sql_worker
    db.init_app(app)
    sql_worker = SQLWorker(app)  # Has to be there

import json
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from .utils import LayersConnectionStatus, DeleteStatus

db = SQLAlchemy()  # Has to be global by Flask documentation


class User(db.Model):  # type: ignore
    __tablename__ = "users_table"

    id = db.Column(db.Integer, primary_key=True)
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
            new_user = User()
            db.session.add(new_user)
            db.session.commit()
            return new_user.id

    def add_model(self, user_id: int, name: str):
        with current_app.app_context():
            model_owner = User.query.filter(User.id == user_id).first()
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
            model = Model.query.filter(Model.id == model_id).first()
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

    def add_connection(
        self, layer_from: int, layer_to: int, model_id: int
    ):  # When we sure, that adding is correct
        with current_app.app_context():
            model = Model.query.filter(Model.id == model_id).first()
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
            model = Model.query.filter(Model.id == model_id).first()
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
            model = Model.query.filter(Model.id == model_id).first()
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

    def check_dimensions(self, layer_from: dict, layer_to: dict):
        return True

    def verify_connection(
        self, user_id: int, model_id: int, layer_from: int, layer_to: int
    ):
        with current_app.app_context():
            if not self.verify_access(user_id, model_id):
                return LayersConnectionStatus.AccessDenied
            model = Model.query.filter(Model.id == model_id).first()
            model_items = json.loads(model.content)
            layers = model_items["layers"]
            layer1 = list(filter(lambda layer: layer["id"] == layer_from, layers))
            layer2 = list(filter(lambda layer: layer["id"] == layer_to, layers))
            if not layer1 or not layer2:
                return LayersConnectionStatus.DoNotExist
            if not self.check_dimensions(layer1[0], layer2[0]):
                return LayersConnectionStatus.DimensionsMismatch
            return LayersConnectionStatus.OK

    def verify_access(self, user_id, model_id):
        with current_app.app_context():
            model_passport = Model.query.filter(Model.id == model_id).first()
            if not model_passport or model_passport.owner != user_id:
                return False
            return True

    def get_graph_elements(self, model_id):
        with current_app.app_context():
            model = Model.query.filter(Model.id == model_id).first()
            if not model:
                return -1
            model_items = json.loads(model.content)
            return model_items

    def train_model(self, model_id: int):
        with current_app.app_context():
            model = Model.query.filter(Model.id == model_id).first()
            if not model:
                return -1
            model.is_trained = True
            db.session.add(model)
            db.session.commit()
            return 1

    def is_model_trained(self, model_id: int):
        with current_app.app_context():
            model = Model.query.filter(Model.id == model_id).first()
            if not model:
                return False
            return model.is_trained


sql_worker: SQLWorker = None  # type: ignore


def init_app(app):
    global sql_worker
    db.init_app(app)
    sql_worker = SQLWorker(app)  # Has to be there

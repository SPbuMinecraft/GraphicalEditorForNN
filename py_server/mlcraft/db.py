from threading import Lock  # fuck it, just using old robust methods
import datetime
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
    models = db.relationship("Model", backref="user_id")


class Model(db.Model):  # type: ignore
    __tablename__ = "models_table"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text)
    owner = db.Column(db.Integer, db.ForeignKey("users_table.id"), nullable=False)
    content = db.Column(db.Text)
    is_trained = db.Column(db.Boolean)
    raw = db.Column(db.Text)


class Metrics(db.Model):  # type: ignore
    __tablename__ = "metrics_table"

    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.Integer, db.ForeignKey("users_table.id"), nullable=False)
    label = db.Column(db.Text)
    values = db.Column(db.Text)
    begin_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    protected = db.Column(db.Boolean)


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
                "Wrong password", HTTPStatus.UNAUTHORIZED, problemPart="password"
            )
        return existing_user.id

    def get_models_list(self, user_id: int) -> list[dict[str, int | str]]:
        user = db.session.get(User, user_id)
        if user is None:
            raise Error("User not found", HTTPStatus.UNAUTHORIZED)
        # only return model ids where 'raw' data is filled
        return list(
            map(
                lambda m: {"id": m.id, "name": m.name},
                filter(lambda m: m.raw, user.models),
            )
        )

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
                raw="",
            )
            db.session.add(new_model)
            db.session.commit()
            return new_model.id

    def get_raw_model(self, model_id: int) -> dict:
        with current_app.app_context():
            model = db.session.get(Model, model_id)
            if not model:
                raise Error(f"No model with id {model_id}")
            return json.loads(model.raw)

    def update_model(
        self, model_id: int, name: str | None = None, raw: str | None = None
    ):
        with current_app.app_context():
            model = self.get_model(model_id)
            if name is not None:
                model.name = name
            if raw is not None:
                model.raw = raw
            db.session.commit()

    def copy_model(self, model_id: int, dst_model_id: int | None = None):
        with current_app.app_context():
            model = self.get_model(model_id)
            new_model = Model(
                owner=model.owner,
                name=model.name,
                content=model.content,
                is_trained=model.is_trained,
                raw="",
            )
            if dst_model_id is not None:
                old_model = db.session.get(Model, dst_model_id)
                if not old_model:
                    raise Error(
                        f"No model with id {dst_model_id}", HTTPStatus.NOT_FOUND
                    )
                old_model.content = new_model.content
                old_model.is_trained = new_model.is_trained
                db.session.add(old_model)
            else:
                db.session.add(new_model)
            db.session.commit()
            return new_model.id

    def delete_model(self, model_id: int):
        with current_app.app_context():
            model = self.get_model(model_id)
            db.session.delete(model)
            db.session.commit()

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

    def update_metrics(self, model_id, values: list[float], label: str, rewrite: bool):
        with current_app.app_context(), self.content_lock:
            metrics = Metrics.query.filter_by(model=model_id, label=label)\
                                    .order_by(Metrics.id.desc()).first()
            if not metrics:
                metrics = Metrics()
                metrics.model = model_id
                metrics.label = label
                metrics.values = " ".join(list(map(str, values)))
                metrics.begin_time = datetime.datetime.now()
                metrics.end_time = datetime.datetime.now()
                metrics.protected = False
            if rewrite:
                metrics.values = ""
            else:
                metrics.values += " "
            metrics.values += " ".join(list(map(str, values)))
            metrics.end_time = datetime.datetime.now()
            db.session.add(metrics)
            db.session.commit()
            return 0

    def protect_metrics(self, model_id, label: str, protected: bool):
        with current_app.app_context(), self.content_lock:
            metrics = Metrics.query.filter_by(model=model_id, label=label)\
                                    .order_by(Metrics.id.desc()).first()
            if not metrics:
                raise Error(
                    f"No recordings found for model with id {model_id} and label {label}.",
                    HTTPStatus.NOT_FOUND,
                )
            metrics.protected = protected
            db.session.add(metrics)
            db.session.commit()
            return 0

    def get_metrics(self, model_id, label: str) -> str:
        with current_app.app_context(), self.content_lock:
            metrics = Metrics.query.filter_by(
                model=model_id, label=label
            ).order_by(Metrics.id.desc()).first()
            if not metrics:
                raise Error(
                    f"No recordings found for model with id {model_id} and label {label}.",
                    HTTPStatus.NOT_FOUND,    
                )
            return metrics.values

    # Пока без ручки, просто как напоминание о том, что метрики нужно чистить
    def delete_old_metrics():
        with current_app.app_context(), self.content_lock:
            Metrics.query.filter(
                Metrics.end_time < datetime.datetime.now() - datetime.timedelta(days=30)
            ).delete()
            Metrics.query.filter(
                Metrics.end_time < datetime.datetime.now() - datetime.timedelta(days=1)
            ).filter_by(protected=False).delete()
            db.session.commit()


sql_worker: SQLWorker = None  # type: ignore


def init_app(app):
    global sql_worker
    db.init_app(app)
    sql_worker = SQLWorker(app)  # Has to be there

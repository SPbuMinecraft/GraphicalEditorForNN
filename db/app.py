import os
import sys

from flask import Flask, jsonify, abort, request, Response
from flask_sqlalchemy import SQLAlchemy
from csv import Error as CSVError
from http import HTTPStatus

import inspect

# Maybe there is a normal solution, who knows??
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import LayersConnectionStatus, parse_parameters, error


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)  # Has to be global by Flask documentation
app.config.from_pyfile("../config.py")

# Needs to be defined in this file for the correct path
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'database.db')

db = SQLAlchemy(app)  # Has to be global by Flask documentation


class User(db.Model):
    __tablename__ = "users_table"

    id = db.Column(db.Integer, primary_key=True)
    model = db.relationship('Model', backref='user_id')


class Model(db.Model):
    __tablename__ = "models_table"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text)
    owner = db.Column(db.Integer, db.ForeignKey('users_table.id'), nullable=False)
    layers = db.relationship('Layer', backref='model_id')


class Layer(db.Model):
    __tablename__ = "layers_table"

    id = db.Column(db.Integer, primary_key=True)
    layer_type = db.Column(db.Text)
    parameters = db.Column(db.Text)
    model = db.Column(db.Integer, db.ForeignKey('models_table.id'), nullable=False)


class Connection(db.Model):
    __tablename__ = "connections_table"

    id = db.Column(db.Integer, primary_key=True)
    layer_from = db.Column(db.Integer, db.ForeignKey('layers_table.id'))
    layer_to = db.Column(db.Integer, db.ForeignKey('layers_table.id'))
    layers_from = db.relationship('Layer', backref='from_id', foreign_keys=[layer_from])
    layers_to = db.relationship('Layer', backref='to_id', foreign_keys=[layer_to])


class SQLWorker:
    def __init__(self):
        with app.app_context():
            db.create_all()

    def add_user(self, user_parameters: dict):
        with app.app_context():
            new_user = User()
            db.session.add(new_user)
            db.session.commit()
            return new_user.id

    def add_model(self, user_id: int):
        with app.app_context():
            model_owner = User.query.filter(User.id == user_id).first()
            if not model_owner:
                return -1
            new_model = Model(owner=user_id)
            db.session.add(new_model)
            db.session.commit()
            return new_model.id

    def add_layer(self, layer_type: str, parameters: str, model_id: int):
        with app.app_context():
            new_layer = Layer(layer_type=layer_type, parameters=parameters, model=model_id)
            db.session.add(new_layer)
            db.session.commit()
            return new_layer.id

    def add_connection(self, layer_from: int, layer_to: int):
        with app.app_context():
            new_conn = Connection(layer_from=layer_from, layer_to=layer_to)
            db.session.add(new_conn)
            db.session.commit()
            return new_conn.id

    def get_model_layers(self, model: int):
        with app.app_context():
            layers = Layer.query.filter(Layer.model == model).all()
            return layers

    def get_model_connections(self, model: int):
        with app.app_context():
            connections_blocks = [Connection.query.filter(Connection.layer_from == layer.id).all() for layer in
                                    Layer.query.filter(Layer.model == model).all()]
            connections = []
            for block in connections_blocks:
                connections += block
            return connections

    def check_dimensions(self, layer_from: Layer, layer_to: Layer):
        return True

    def verify_connection(self, user_id: int, layer_from: int, layer_to: int):
        with app.app_context():
            layer1 = Layer.query.filter(Layer.id == layer_from).first()
            layer2 = Layer.query.filter(Layer.id == layer_to).first()
            if not layer1 or not layer2:
                return LayersConnectionStatus.DoNotExist
            if layer1.model != layer2.model:
                return LayersConnectionStatus.FromDifferentModels
            if not self.verify_access(user_id, layer1.model):
                return LayersConnectionStatus.AccessDenied
            if not self.check_dimensions(layer1, layer2):
                return LayersConnectionStatus.DimensionsMismatch
            return LayersConnectionStatus.OK

    def verify_access(self, user_id, model_id):
        with app.app_context():
            model_passport = Model.query.filter(Model.id == model_id).first()
            if not model_passport or model_passport.owner != user_id:
                return False
            return True


sql_worker = SQLWorker()  # Has to be there    


if __name__ == "__main__":
    app.run(host="localhost", port=4000, debug=True)

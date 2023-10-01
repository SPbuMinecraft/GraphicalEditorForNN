import os

from flask import Flask, jsonify, abort, request, Response
from flask_sqlalchemy import SQLAlchemy
from csv import Error as CSVError
from http import HTTPStatus


def error(code: int, message: str):
    abort(Response(message, code))


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)  # Has to be global by Flask documentation
app.config['SECRET_KEY'] = 'minecraft'  # Setting config
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)  # Has to be global by Flask documentation


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    models = db.relationship('Model', backref='user_id')


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text)
    owner = db.Column(db.Integer, db.ForeignKey('user.id'))
    layers = db.relationship('Layer', backref='model_id')


class Layer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.Text)
    parameters = db.Column(db.Text)
    model = db.Column(db.Integer, db.ForeignKey('model.id'))


class Connection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    layer_from = db.Column(db.Integer, db.ForeignKey('layer.id'))
    layer_to = db.Column(db.Integer, db.ForeignKey('layer.id'))
    layers_from = db.relationship('Layer', backref='from_id', foreign_keys=[layer_from])
    layers_to = db.relationship('Layer', backref='to_id', foreign_keys=[layer_to])


class SQLWorker:
    def __init__(self):
        with app.app_context():
            db.create_all()

    def add_layer(self, type: str, parameters: str, model: int):
        with app.app_context():
            db.session.add(Layer(type=type, parameters=parameters, model=model))
            db.session.commit()

    def add_connection(self, layer_from: int, layer_to: int):
        with app.app_context():
            if self.check_dimensions(layer_from, layer_to):
                db.session.add(Connection(layer_from=layer_from, layer_to=layer_to))
                db.session.commit()

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

    def check_dimensions(self, layer_from, layer_to):
        return True


sql_worker = SQLWorker()  # Has to be there


@app.route('/add_layer', methods=['POST'])
def add_layer():
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        sql_worker.add_layer(json['type'], json['parameters'], json['model'])
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    except CSVError as e:
        error(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))
    return "done", HTTPStatus.CREATED


@app.route('/add_connection', methods=['POST'])
def add_connection():
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        sql_worker.add_connection(json['layer_from'], json['layer_to'])
    except KeyError as e:
        error(HTTPStatus.BAD_REQUEST, message=str(e))
    except CSVError as e:
        error(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))
    return "done", HTTPStatus.CREATED


@app.route('/get_graph_elements/<int:model_id>', methods=['GET'])
def get_graph_elements(model_id: int):
    layers = [{
        'type': layer.type,
        'parameters': layer.parameters,
    } for layer in sql_worker.get_model_layers(model_id)]
    connections = [{
        'layer_from': connection.layer_from,
        'layer_to': connection.layer_to
    } for connection in sql_worker.get_model_connections(model_id)]
    return jsonify({'layers': layers, 'connections': connections})


if __name__ == "__main__":
    app.run(host="localhost", port=4000, debug=True)

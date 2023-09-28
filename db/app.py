import os

from flask import Flask, jsonify, abort, request, Response
from flask_sqlalchemy import SQLAlchemy
from csv import Error as CSVError
from http import HTTPStatus


def error(code: int, message: str):
    abort(Response(message, code))


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'minecraft'
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    layers = db.relationship('Layer', backref='user')


class Layer(db.Model):
    __tablename__ = 'layer'
    id = db.Column(db.Integer, primary_key=True)
    in_features = db.Column(db.Integer, nullable=False)
    out_features = db.Column(db.Integer, nullable=False)
    bias = db.Column(db.Boolean, nullable=False)
    owner = db.Column(db.Integer, db.ForeignKey('user.id'))


class Connection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    layer_from = db.Column(db.Integer, db.ForeignKey('layer.id'))
    layer_to = db.Column(db.Integer, db.ForeignKey('layer.id'))
    layers_from = db.relationship('Layer', backref='from', foreign_keys=[layer_from])
    layers_to = db.relationship('Layer', backref='to', foreign_keys=[layer_to])


class SQLWorker:
    def __init__(self):
        with app.app_context():
            db.create_all()

    def add_layer(self, in_features: int, out_features: int, bias: bool, owner: int):
        with app.app_context():
            db.session.add(Layer(in_features=in_features, out_features=out_features, bias=bias, owner=owner))
            db.session.commit()

    def add_connection(self, layer_from: int, layer_to: int):
        with app.app_context():
            if self.check_dimencions(layer_from, layer_to):
                db.session.add(Connection(layer_from=layer_from, layer_to=layer_to))
                db.session.commit()

    def get_user_layers(self, owner: int):
        with app.app_context():
            layers = Layer.query.filter(Layer.owner == owner).all()
            return layers

    def get_user_connections(self, owner: int):
        with app.app_context():
            connections_blocks = [Connection.query.filter(Connection.layer_from == layer.id).all() for layer in
                                    Layer.query.filter(Layer.owner == owner).all()]
            connections = []
            for block in connections_blocks:
                connections += block
            return connections

    def check_dimencions(self, layer_from, layer_to):
        return True


sql_worker = SQLWorker()


@app.route('/add_layer', methods=['POST'])
def add_layer():
    json = request.json
    if not json:
        error(HTTPStatus.BAD_REQUEST, message="No json provided")
    try:
        sql_worker.add_layer(json['in_features'], json['out_features'], json['bias'], json['owner'])
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


@app.route('/get_graph_elements/<int:user_id>', methods=['GET'])
def get_graph_elements(user_id: int):
    layers = [{
        'in_features': layer.in_features,
        'out_features': layer.out_features,
        'bias': layer.bias
    } for layer in sql_worker.get_user_layers(user_id)]
    connections = [{
        'layer_from': connection.layer_from,
        'layer_to': connection.layer_to
    } for connection in sql_worker.get_user_connections(user_id)]
    return jsonify({'layers': layers, 'connections': connections})


if __name__ == "__main__":
    app.run(host="localhost", port=4000, debug=True)

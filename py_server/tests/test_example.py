import json
from http import HTTPStatus
from flask import Flask
from flask.testing import FlaskClient
from mlcraft.db import Model


def post_request(client: FlaskClient, url: str, data: dict):
    return client.post(url, data=json.dumps(data), content_type="application/json")


def test_add_user(client: FlaskClient):
    response = post_request(
        client,
        "/user",
        data={"login": "biba", "mail": "boba@mail.ru", "password": "bibaboba"},
    )
    assert response.status_code == HTTPStatus.CREATED
    assert response.data == b'{"user_id":3}\n'


def test_add_model(client: FlaskClient, app: Flask):
    post_request(
        client,
        "/user",
        data={"login": "biba", "mail": "boba@mail.ru", "password": "bibaboba"},
    )
    response = post_request(client, "/model/3", data={"name": "My fresh AI"})
    assert response.status_code == HTTPStatus.CREATED
    assert response.data == b'{"model_id":5}\n'
    with app.app_context():
        m = Model.query.filter(Model.id == 5).first()
    assert m.id == 5
    assert m.owner == 3
    assert m.is_trained == False
    assert m.name == "My fresh AI"

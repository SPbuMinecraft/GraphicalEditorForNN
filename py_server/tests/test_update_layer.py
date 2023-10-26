import json
from http import HTTPStatus
from conftest import db, Model
from flask import Flask
from flask.testing import FlaskClient
from test_example import post_request


def put_request(client: FlaskClient, url: str, data: dict):
    return client.put(url, data=json.dumps(data), content_type="application/json")


def test_update_layer(client: FlaskClient, app: Flask):
    r = post_request(
        client,
        "/add_layer/1/1",
        data={"type": "Linear", "parameters": {"inFeatures": 2, "outFeatures": 1}},
    )
    assert r.status_code == HTTPStatus.CREATED
    r = put_request(
        client,
        "/update_layer/1/1",
        data={"id": 0, "parameters": {"inFeatures": 3, "outFeatures": 100}},
    )
    assert r.status_code == HTTPStatus.OK
    with app.app_context():
        m = db.session.get(Model, 1)
    assert m is not None
    params = json.loads(m.content)["layers"][0]["parameters"]
    assert params["inFeatures"] == 3
    assert params["outFeatures"] == 100

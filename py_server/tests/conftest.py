import shutil
import tempfile
import pytest
from flask import Flask

from mlcraft import make_app
from mlcraft.db import db, User, Model

empty = '{"connections": [], "layers": []}'
data = [
    ("My imaginary OR", 1, empty, False),
    ("My imaginary AND", 1, empty, True),
    ("My imaginary XOR", 2, empty, False),
    ("My imaginary NAND", 2, empty, True),
]


def model_from(entry: tuple[str, int, str, bool]) -> Model:
    return Model(name=entry[0], owner=entry[1], content=entry[2], is_trained=entry[3])


@pytest.fixture
def app():
    # create a temporary file to isolate the database for each test
    path = tempfile.mkdtemp()
    app = make_app(
        {
            "TESTING": True,  # flask changes some internal behavior so it’s easier to test
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{path}/database.db",
        }
    )

    with app.app_context():
        db.session.add(User())
        db.session.add(User())
        for entry in data:
            db.session.add(model_from(entry))
        db.session.commit()

    yield app

    shutil.rmtree(path)


@pytest.fixture
def client(app: Flask):
    return app.test_client()

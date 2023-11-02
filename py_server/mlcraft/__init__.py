import os
import json
from flask import Flask
from flask_cors import CORS


def extract_config(file):
    configs = json.load(file)
    config = configs["py_server"]

    cpp = configs["cpp_server"]
    config["CPP_SERVER"] = f"http://{cpp['HOST']}:{cpp['PORT']}"

    client = configs["client"]
    config["CLIENT"] = f"http://{client['HOST']}:{client['PORT']}"

    return config


def make_app(config=None):
    app = Flask(__name__)
    # create a folder where the application will be launched
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError as e:
        print(f"Failed to create an instance folder")
        raise e

    default_path = "../../config.json"
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        print(f"CONFIG_PATH variable not set, using default path: {default_path}")
        config_path = default_path

    app.config.from_file(config_path, load=extract_config)
    if config is not None:
        app.config.update(config)

    CORS(app, origins=app.config["CLIENT"])
    # uncomment this and comment line above if you want to make it simple
    # CORS(app)

    from . import db  # this is ok, but only for professional programmers
    db.init_app(app)

    from . import server
    app.register_blueprint(server.app)

    return app

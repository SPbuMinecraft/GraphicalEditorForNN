import os
import json
from flask import Flask


def extract_config(file):
    configs = json.load(file)
    config = configs["py_server"]

    cpp = configs["cpp_server"]
    config["CPP_SERVER"] = f"http://{cpp['HOST']}:{cpp['PORT']}"

    return config


def make_app(config=None):
    app = Flask(__name__)
    # create a folder where the application will be launched
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError as e:
        print(f"Failed to create an instance folder")
        raise e

    app.config.from_file("../../config.json", load=extract_config)
    if config is not None:
        app.config.update(config)

    from . import db  # this is ok, but only for professional programmers
    db.init_app(app)

    from . import server
    app.register_blueprint(server.app)

    return app
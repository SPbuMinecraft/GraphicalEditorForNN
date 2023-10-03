import os
from utils import to_port


PY_SERVER_PORT = to_port(2000)
# CLIENT_PORT = to_port(2000)
# SERVER_PORT = to_port(3000)
DB_PORT = to_port(4000)

CLIENT_HOSTNAME = "localhost"
SERVER_HOSTNAME = "localhost"
PY_SERVER_HOSTNAME = "localhost"
DB_HOSTNAME = "localhost"

SECRET_KEY = "minecraft"
SQLALCHEMY_TRACK_MODIFICATIONS = False

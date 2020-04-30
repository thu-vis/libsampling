from flask import Flask
import os
import sys

SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)

app = Flask(__name__, static_url_path="/static")

if __name__ == '__main__':
    app.run(port=8085, host="0.0.0.0", threaded=True)

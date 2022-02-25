import flask

app = flask.Flask(__name__)
#This switch the debug mode (auto reload). True = debug on; False = debug off
app.config["DEBUG"] = False

from routes import *
app.register_blueprint(routes)

if __name__ == '__main__':
    app.run()
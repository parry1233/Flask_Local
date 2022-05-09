import flask

app = flask.Flask(__name__)
#This switch the debug mode (auto reload). True = debug on; False = debug off
app.config["DEBUG"] = False
app.config["FOM_RESULT"] = 'machineLearning/first-order-model/results'
app.config["CROP"] = 'machineLearning/first-order-model/driving_video'

from routes import *
app.register_blueprint(routes)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8888)
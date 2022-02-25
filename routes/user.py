from flask import jsonify, request
from . import routes

@routes.route("/",methods=['GET'])
def welcome():
    return "<h1>Hello World</h1>"

@routes.route("/Hello",methods=['POST'])
def postwelcome():
    return "<h1>Hello PostMAAAN</h1>"
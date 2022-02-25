from flask import Blueprint

routes = Blueprint('routes', __name__)
from .user import *
from .machine import *
# from .Order import *
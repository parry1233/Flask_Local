from flask import jsonify, request
from . import routes
from machineLearning import tf_version
from machineLearning.PredictSock import PredictStock as PStock
from machineLearning.Categorical_Classify.Multiple_Class_CNN import MCNN

@routes.route("/mltry",methods=['GET'])
def mltry():
    one, two = tf_version.tfversion()
    print(one,two)
    return '<h1>'+one+'</h1><h1>'+str(two)+'</h1>'

@routes.route("/stock",methods=['POST'])
def learn():
    sid = request.get_json()['sid']
    pstock = PStock()
    check = pstock.train(sid)
    #print(sid)
    
    return jsonify([
        {
            'sid': sid,
            'status': check
        }
    ])
    
@routes.route("/stockPredict",methods=['POST'])
def predict():
    sid = request.get_json()['sid']
    pstock = PStock()
    pstock.train(sid)
    price, status = pstock.model_predict()
    
    return jsonify([
        {
            'sid': sid,
            "price": price.tolist(),
            'status': status
        }
    ])
    
@routes.route("/MCNN",methods=['POST'])
def mcnn():
    #sid = request.get_json()['sid']
    m_cnn = MCNN()
    data = m_cnn.run()
    
    return jsonify([
        {
            'data': data
        }
    ])
    
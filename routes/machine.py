from fileinput import filename
import io
import numpy as np
import base64
import os, glob
from os import abort
from flask import jsonify, request, send_from_directory, current_app
from . import routes
from machineLearning import tf_version
from machineLearning.PredictSock import PredictStock as PStock
from machineLearning.Categorical_Classify.Multiple_Class_CNN import MCNN
#from machineLearning import first_order_model
import subprocess
from PIL import Image

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
    
@routes.route("/FOMupload",methods=['POST'])
def fom():
    if not request.get_json()['img']:
        abort(400)
    
    # get imageName
    img_name = request.get_json()['name']
    # get cropType
    crop_name = request.get_json()['crop']
    # get the base64 encoded string
    img_b64 = request.get_json()['img']
    # convert it into bytes
    img_bytes = base64.b64decode(img_b64.encode('utf-8'))
    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)
    img = img.save('machineLearning/first-order-model/source_image/'+img_name)
    
    
    FOMprocess = subprocess.Popen(['python','machineLearning/first-order-model/demo.py', 
                                   '--config', 'machineLearning/first-order-model/config/vox-adv-256.yaml',
                                   '--driving_video', 'machineLearning/first-order-model/driving_video/'+crop_name+'.mp4',
                                   '--source_image', 'machineLearning/first-order-model/source_image/'+img_name,
                                   '--checkpoint', 'machineLearning/first-order-model/fom_checkpoints/vox-adv-cpk.pth.tar',
                                   '--result_video', 'machineLearning/first-order-model/results/result.mp4',
                                   '--relative', '--adapt_scale'])
    FOMprocess.wait()
    
    return jsonify([
        {
            'status': 'Done'
        }
    ])

@routes.route("/FOMresult",methods=['GET','POST'])
def getFom():
    if request.method == 'GET':
        filename = 'result.mp4'
        uploads = current_app.config["FOM_RESULT"]
        try:
            return send_from_directory(directory=uploads,path=filename)
        except FileNotFoundError:
            abort(400)
        except:
            abort(404)
    elif request.method =='POST':
        if not request.get_json()['filename']:
            abort(400)
        filename = request.get_json()['filename']
        #filename = 'result.mp4'
        uploads = current_app.config["FOM_RESULT"]
        try:
            return send_from_directory(directory=uploads,path=filename)
        except FileNotFoundError:
            abort(400)
        except:
            abort(404)
    else:
        abort(404)
        
@routes.route("/Crop",methods=['GET'])
def getCrop():
    filename = request.args.get('type')+'.mp4'
    uploads = current_app.config["CROP"]
    try:
        return send_from_directory(directory=uploads,path=filename)
    except FileNotFoundError:
        abort(400)
    except:
        abort(404)

@routes.route("/AllCrop", methods=['GET'])
def getAllCrop():
    try:
        path = 'machineLearning\\first-order-model\\driving_video'
        videoList = []
        for filename in glob.glob(path+'\\*.mp4'):
            videoList.append(os.path.splitext(os.path.basename(filename))[0])
        return jsonify([
            {
                'all': videoList
            }
        ])
    except: abort(404)

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\Maneesha\\Cricket Object Detection\\Model_Resnet50.hdf5'
# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    #x=x/255.
    #x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    x=np.expand_dims(x, axis=0)
    x=preprocess_input(x, mode='caffe')
    preds=model.predict(x)
    return(preds)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    

    #preds = model.predict(x)
    #preds=np.argmax(preds, axis=1)
    #if preds==0:
      #  preds="BALL"
    #elif preds==1:
     #   preds="BAT"
    #elif preds==2:
        #preds="FIELD"
    #elif preds==3:
        #preds="GROUNDSTAND"
    #elif preds==4:
        #preds="PITCH"
    #elif preds==5:
        #preds="SCORE"
    #elif preds==6:
        #preds="UMPIRE"
    #else:
     #   preds="WICKET"
    
    
    #return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

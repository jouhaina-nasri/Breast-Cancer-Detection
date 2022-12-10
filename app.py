from flask import Flask, render_template,request,flash,redirect

import tensorflow as tf
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing.image import ImageDataGenerator
import uvicorn 
import numpy as np

import keras.utils as image



class Image(BaseModel):
    imageName : str
        
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 128,
                                                 class_mode = 'categorical',
                                                 shuffle=True)
#model loading
cnn=tf.keras.models.load_model("./myModel_v1")


app = FastAPI()


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app=Flask(__name__)

#-------------------------------------------------------------------------------------------------------#
# page d'accueil #
@app.route('/')
def accueil ():
    return render_template('index.html')
###########

# test #
@app.route('/test')
def get_image_category():
    return render_template('test.html')

from werkzeug.utils import secure_filename

import PIL.Image
import cv2

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file'].read()
      npimg = np.fromstring(f,np.uint8)
      img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
      print(img)
      test_image = image.img_to_array(img)
      test_image = np.expand_dims(test_image, axis = 0)
      print(test_image) 
      result = cnn.predict(test_image)
      training_set.class_indices
      print(result)
      if result[0][0] == 0:
        prediction = 'Cancer'
      else:
        prediction = 'NORMAL'
      return {'classificaton_result': prediction}
     

#main#
if __name__=="__main__":
    app.run(debug=True)
#------------------------#

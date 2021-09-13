from flask import Flask,render_template,request
import keras
from keras.saving.saved_model.json_utils import decode
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
app = Flask(__name__)
@app.route('/',methods=['GET'])
def demo():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def predict():
    img = request.files['img']
    img_path = "./images/"+img.filename
    img.save(img_path)

    model1 = load_model("./best.h5")
    img = load_img(path=img_path,target_size=(224,224))
    arr = img_to_array(img)/255
    print(arr.shape)
    arr = np.expand_dims(arr,axis=0)
    pred = model1.predict(arr)[0][0]
    str = f"The chance of tumor is {pred*100}%"
    print(str)

    return render_template("index.html",prediction=str)

if __name__ == '__main__':
    app.run(port=3000,debug=True)
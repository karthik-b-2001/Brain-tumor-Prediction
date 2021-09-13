import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
model1 = load_model("./best.h5")
img = load_img(path="./no.jpg",target_size=(224,224))
arr = img_to_array(img)/255
print(arr.shape)
arr = np.expand_dims(arr,axis=0)
pred = model1.predict(arr)[0][0]
print(f"The chance of tumor is {pred*100}%")
plt.imshow(img)
plt.show()
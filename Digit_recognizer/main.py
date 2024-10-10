import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from data_load import load_image
from preprocess import tensor_to_gscale,normalizing
from model import model_building,compile_model,find_loss

(x_train, y_train), (x_test, y_test) = load_image()



def printing(images):
 plt.imshow(images[3])
images = tensor_to_gscale(x_train)
printing(images)

x_train,x_test = normalizing(x_train,x_test)





model = model_building()
model = compile_model(model)
model.fit(x_train,y_train,epochs = 20,batch_size = 24,validation_data =(x_test,y_test))






find_loss(model,x_test,y_test)


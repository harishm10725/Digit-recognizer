from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np


def tensor_to_gscale(x_train):
  img_from_array = []
  for i in range(len(x_train)):
    image_array = x_train[i].reshape(x_train[i].shape[0],x_train[i].shape[1],1)
    img = array_to_img(image_array)
    img = img.convert('L')
    img_from_array.append(np.array(img))
  return img_from_array


def normalizing(x_train,x_test):
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  return x_train,x_test

from tensorflow.keras.datasets import mnist as dataset
def load_image():
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    return (x_train, y_train), (x_test, y_test)


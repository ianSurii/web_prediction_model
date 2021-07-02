from os import listdir
from PIL import Image as PImage
import PIL
import PIL.Image
from tensorflow import keras
import pathlib
from tensorflow.keras import layers
import matplotlib.pyplot as plt
base = "../input/brest-cancer/Breast Cancer DataSet/"
train_base="Train/"
test_base="Test/"
data_dir = pathlib.Path('../input/brest-cancer/Breast Cancer DataSet/Train/')
test_data_dir = pathlib.Path('../input/brest-cancer/Breast Cancer DataSet/valid/')


PATH_TO_CKPT = '../saved_model.pb'
def load_image(image):
    # image_path="../input/brest-cancer/Breast Cancer DataSet/Test/Malignant/SOB_M_DC-14-12312-400-019.png"
    # img = keras.preprocessing.image.load_img(
    # image_path, target_size=(img_height, img_width)
    # )
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

   result=
    """This Breast has {} conditions with an {:.2f} percent confidence.
    
    Note Benign is not cancers cell while malignant is cancer cell
    """
    .format(class_names[np.argmax(score)], 100 * np.max(score)
    return result





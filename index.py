import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np 
import pandas as pd
import os

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


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input image file", type=["png","jpg","svg"])
if uploaded_file is not None:
    print("0")

   

else:
    
    def user_input_features():
        # data_dir = pathlib.Path('../input/brest-cancer/Breast Cancer DataSet/Train/')
        # files= list(data_dir.glob('*/*.png'))
        island = st.sidebar.selectbox('Images',('Biscoe','Dream','Torgersen'))

        
      
        features=""
        return features
    # input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
# breast= pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'], axis=1)
# df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    # st.write(uploaded_file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    model = tf.keras.models.load_model("/home/astra/Vscode/web_prediction_model/saved_model.pb")
    st.write(model.summary)
    # load_image(uploaded_file)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write()



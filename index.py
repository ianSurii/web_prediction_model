import streamlit as st
import numpy as np
import tensorflow as tf
import os
from os import listdir
import PIL
import PIL.Image
from tensorflow import keras
import pathlib
import datetime;






from os import listdir

from tensorflow import keras


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
                visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("UPLOAD IMAGE FILE", type=["png","jpg","svg"])
if uploaded_file is not None:
    print("0")

   

else:
    
    def user_input_features():
        data_dir = pathlib.Path('model/images/')
        # data_dir = pathlib.Path('../input/brest-cancer/Breast Cancer DataSet/Train/')
        # files= list(data_dir.glob('*/*.png'))

        
        island = st.sidebar.selectbox('Images',('model/images/examples/217516.png','model/images/examples/396561.png','model/images/examples/515121.png'))

        data = {'images': island}
      
        
        return data
    image_location_and_name=user_input_features()
    # image_location_and_name
    st.image(image_location_and_name['images'])
    
    PIL.Image.open(str(image_location_and_name['images']))

    img = keras.preprocessing.image.load_img(image_location_and_name['images'], target_size=(180, 180))
    
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    class_names = ["Bening","Malignant"]
    reconstructed_model = keras.models.load_model(os.path.join("model"))

# Let's check:
    # try:


    savedmodel=reconstructed_model.predict(img_array)
    score = tf.nn.softmax(savedmodel[0])

    result="""
    This Breast has {} conditions with an {:.2f} percent confidence.
    
    Note Benign is not cancers cell while malignant is cancer cell
    """.format(class_names[np.argmax(score)], 100 * np.max(score))
    st.write(result)




if uploaded_file is not None:
    
    ts = datetime.datetime.now().timestamp()
    file_name=str(ts)+'.png'
  
    image_location_and_name='tempDir/'+str(ts)+'.png'
    image_file=pathlib.Path(image_location_and_name)
   
    

    with open(os.path.join("tempDir/",file_name),"wb") as f:
         f.write(uploaded_file.getbuffer())

  
    class_names = ["Bening","Malignant"]
    
  

    img = keras.preprocessing.image.load_img(image_location_and_name, target_size=(180, 180))
    
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    

    reconstructed_model = keras.models.load_model(os.path.join("model"))

# Let's check:
    # try:


    savedmodel=reconstructed_model.predict(img_array)
    score = tf.nn.softmax(savedmodel[0])

    result="""
    This Breast has {} conditions with an {:.2f} percent confidence.
    
    Note Benign is not cancers cell while malignant is cancer cell
    """.format(class_names[np.argmax(score)], 100 * np.max(score))
    st.subheader    (result)
    st.image(uploaded_file, channels="BGR")
    # except:


    
else:
    st.write('Awaiting image file to be uploaded. Currently using example input parameters (shown below).')
    st.write()

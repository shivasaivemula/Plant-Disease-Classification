#importing libraries
import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

st.set_page_config(
     page_title="Plant disease App",
     page_icon="🧊",
     layout="wide"
 )

 
model_path = os.path.abspath(os.path.join(os.getcwd(),  "Plant_disease.h5"))
#loading the model
model= tf.keras.models.load_model(model_path)

import pathlib
import numpy as np
import pandas as pd
rem_path=os.path.abspath(os.path.join(os.getcwd(),  "remidies.csv"))
df_rem=pd.read_csv(rem_path)
class_names=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
df_class_names = pd.DataFrame(class_names, columns =["Disease Classification Classes"])

activities = ["HOME" ,"Plant Disease"]
choice = st.sidebar.selectbox("Select Activty",activities)

def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)	
if choice =='HOME':
    st.title(":blue[Plant Disease Detection & Classification ]")
    st.markdown("This Application makes easy for farmers, biologists, and botanists to identify plant or crop types or type of disease and spot any problems in them. The software uploads the plant image to the server for analysis using the CNN classifier model. When a sickness is found, the user is shown the problem and the Remidies....", unsafe_allow_html=True)
    st.subheader("To predict the Disease of your plant Go to Plant Disease in Dashboard.. 👈🏻")
    #st.caption(":orange[To view my Linkedin Profile click on ]👉🏻 https://www.linkedin.com/in/uday-kiran-4aa25b1b5/")
    st.subheader('Model')
    st.markdown('There are 38 classes for classification of 14 plants')
    st.dataframe(df_class_names,1000,1400)

if choice == 'Plant Disease':
        st.title(":blue[Plant Disease Classification]")
        st.write("Upload your Plant's Leaf Image and get predictions if the plant is healthy or not and know the type of Disease...")
        # Setting the files that can be uploaded
        image_input = st.file_uploader("Upload Image",type=['jpg'])
        st.markdown("* * *")
        

        if image_input is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(image_input)
            image = image.resize((200, 200))
            st.image(image)
            img_array = np.array(image)
            img = tf.image.resize(img_array, size=(256, 256))
            img = img/255.0
            img = tf.expand_dims(img, axis=0)
            st.write("filename:", image_input.name)
            st.markdown("* * *")
            if st.button('Predict'):
                 with st.spinner('Your image is processing'):
                    prediction = model.predict(img)
                    st.success('Done!')
                    st.write(prediction.shape)
                    st.write(prediction)
                    k=class_names[np.argmax(prediction)]
                    a=k.split('___')
                    if a[1]=='healthy':
                        st.write(f"Prediction:\n leave name:{a[0]} \n leave is : {a[1]}")
                    else:
                        b=a[1].split('_')
                        b=' '.join(b)
                        st.title("Prediction :")
                        remd="".join(list(df_rem[df_rem["Diseases"]==k]["Remedies"]))
                        plant=["Leaf name","Leaf is","Disease Name","Remidies"]
                        vals=[a[0],"unhealthy",b,remd]
                        d={"Details":plant,"Values":vals}
                        kd=pd.DataFrame(d)
                        st.table(kd)


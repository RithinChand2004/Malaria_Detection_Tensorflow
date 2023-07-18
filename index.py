#import all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

#Load the model
new_model = tf.keras.models.load_model('model')

col1, col2 = st.columns([1,4], gap="small")

with col1:
    st.image("logo.png", width=100)
    
with col2:    
    st.title("Malaria Detection")


st.subheader("Check if your body cell is infected with malaria or not")

st.image("banner.jpg", use_column_width='auto')

tab1, tab2, tab3 = st.tabs(["About", "Detection", "Our Team"])

with tab1:
    st.title("About the Model")
    
    with st.expander("How to use"):
        st.write("To analyze the malaria infection in your cell, please follow these steps: obtain an image of the cell: Make sure you have a clear image of the cell you wish to analyze. Upload the image: Click on the designated Upload button and select the cell image from your device. Please note that the image format should be in JPG, JPEG, or PNG, and the file size should be less than 200 MB. Processing: Once you've uploaded the image, our highly trained Convolutional Neural Network (CNN) model will begin processing it. Malaria infection detection: The CNN model will carefully examine the image to identify any signs of malaria infection within your cell.")
    with st.expander("Get to know about the model"):
        st.image("models.png", use_column_width='auto')

with tab2:
    #Upload the image
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if image is not None:
        if st.button("Predict"):
            input = tf.keras.utils.load_img(image)
            input = tf.keras.preprocessing.image.img_to_array(input)
            input = np.array([input])
            
            result = new_model.predict(input)
            percentage = int(result[0][0] * 100)
            percentage_str = "{:.2f}%".format(percentage)
            message = "The cell is " + percentage_str + " not infected"
            if percentage == 0:
                
                st.error(message)
                
            elif percentage >= 70:
                
                st.success(message)
            else:
                
                st.warning(message)
            
            plt.pie([percentage, (100-percentage)], labels= ['Not infected', 'Infected'], explode= (0, 0.1), autopct='%1.1f%%', startangle=45, colors=['green', 'red'], shadow=True, )
            st.pyplot(plt)

with tab3:
    st.write("This project is made by:")
    
    df = pd.DataFrame({'Name': ['Surya Vamsi Vema', 'Shinoy Yandra', 'Margana Girish Vardhan', 'V Rithin Chand', 'Rajendra Kumar V', 'Lekha Sathvik D'], 'roll number': ['AM.EN.U4AIE21162', 'AM.EN.U4AIE21168', 'AM.EN.U4AIE21143', 'AM.EN.U4AIE21174', 'AM.EN.U4AIE21164', 'AM.EN.U4AIE21140']})
    
    st.table(df)

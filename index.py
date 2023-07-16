#import all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

#Load the model
new_model = tf.keras.models.load_model('model')

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
       st.write("The cell", percentage_str, "not affected")
       
       plt.pie([percentage, (100-percentage)], labels= ['Not infected', 'Infected'], explode= (0, 0.1), autopct='%1.1f%%', startangle=45, colors=['green', 'red'], shadow=True)
       st.pyplot(plt)



import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Load your pre-trained model
loaded_model_imageNet = load_model("model_resnet50.h5")

# Class names
name_class = ['anemic', 'non-anemic']

# Streamlit title and file uploader
st.title("Anemia Detection")
# Upload image via Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Resize the image to the required input size for the model
    img = cv2.resize(img, (100, 100))
    
    # Convert BGR to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for the model
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    
    # Make predictions using the loaded model
    result = loaded_model_imageNet.predict(x)
    
    # Convert the result to percentage format
    pred_percentages = (result * 100).astype(int)
    
    # Get the predicted class index
    pred_class_index = np.argmax(pred_percentages)
    
    # Display the prediction
    st.write(f"Prediction: {name_class[pred_class_index]} with confidence {pred_percentages[0][pred_class_index]}%")

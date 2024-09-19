import streamlit as st
import numpy as np
import cv2
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io

# Load your pre-trained model
loaded_model_imageNet = load_model("model_resnet50.h5")

# Class names
name_class = ['anemic', 'non-anemic']

# Database setup
conn = sqlite3.connect('anemia_detection.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
          CREATE TABLE IF NOT EXISTS results
          (id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT,
          age INTEGER,
          image BLOB,
          prediction TEXT,
          confidence INTEGER)
          ''')
conn.commit()

# Streamlit title and input fields
st.title("Anemia Detection")

name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=120)

# Upload image via Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None and name and age:
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

    # Save the image as binary for storage in the database
    img_pil = Image.fromarray(img_rgb)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_binary = img_byte_arr.getvalue()

    # Insert the result into the database
    c.execute("INSERT INTO results (name, age, image, prediction, confidence) VALUES (?, ?, ?, ?, ?)",
              (name, age, img_binary, name_class[pred_class_index], int(pred_percentages[0][pred_class_index])))
    conn.commit()
    st.success("Result saved to the database.")

# Option to view saved records
if st.button("View Results"):
    c.execute("SELECT * FROM results")
    rows = c.fetchall()
    
    for row in rows:
        st.write(f"Name: {row[1]}, Age: {row[2]}, Prediction: {row[4]}, Confidence: {row[5]}%")
        img_data = row[3]
        img_display = Image.open(io.BytesIO(img_data))
        st.image(img_display, caption=f"Uploaded by {row[1]}", use_column_width=True)

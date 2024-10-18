import streamlit as st
import sqlite3
import hashlib
from PIL import Image
import numpy as np
import cv2
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load your pre-trained model
loaded_model_imageNet = load_model("model_resnet50.h5")

# Class names
name_class = ['anemic', 'non-anemic']

# Database setup
conn = sqlite3.connect('anemia_detection.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
          CREATE TABLE IF NOT EXISTS users
          (id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE,
          password TEXT)
          ''')

# Create results table if it doesn't exist
c.execute('''
          CREATE TABLE IF NOT EXISTS results
          (id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER,
          name TEXT,
          age INTEGER,
          image BLOB,
          prediction TEXT,
          confidence INTEGER,
          FOREIGN KEY (user_id) REFERENCES users (id))
          ''')
conn.commit()

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to verify password
def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)

# Function for user registration
def register_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()

# Function to authenticate user
def authenticate_user(username, password):
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    if user and verify_password(user[2], password):
        return user[0]  # Return user ID if authentication is successful
    return None

# Function for palm image validation
def validate_palm_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Mask the skin color in the image
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Use contour detection to check for hand shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False, "No palm detected. Please upload a palm image."

    # Check if large enough contour is found (adjust threshold as needed)
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            return True, "Palm detected"
    
    return False, "Invalid image. Please upload a palm image."

# Initialize session state for user login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# Streamlit login page
st.title("Anemia Detection App")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    
    if st.button("Register"):
        try:
            register_user(new_user, new_password)
            st.success("You have successfully created an account")
        except sqlite3.IntegrityError:
            st.error("Username already exists. Please try another.")

elif choice == "Login":
    if not st.session_state.logged_in:
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user_id = authenticate_user(username, password)

            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.success(f"Welcome {username}")
            else:
                st.error("Invalid username or password")
    
    if st.session_state.logged_in:
        # After login, ask for name, age, and image
        st.subheader("Upload Details for Anemia Prediction")

        name = st.text_input("Enter your name")
        age = st.number_input("Enter your age", min_value=0, max_value=120)
        
        # Upload image via Streamlit
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        
        if uploaded_file is not None and name and age:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Validate if the uploaded image is a palm image
            is_valid, message = validate_palm_image(img)
            
            if not is_valid:
                st.error(message)
            else:
                # Resize the image to the required input size for the model
                img_resized = cv2.resize(img, (100, 100))

                if st.button("Submit"):
                    # Convert BGR to RGB for Streamlit display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Display the uploaded image
                    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
                    
                    # Preprocess the image for the model
                    x = np.expand_dims(img_resized, axis=0)
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
                    c.execute("INSERT INTO results (user_id, name, age, image, prediction, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                              (st.session_state.user_id, name, age, img_binary, name_class[pred_class_index], int(pred_percentages[0][pred_class_index])))
                    conn.commit()
                    st.success("Result saved to the database.")




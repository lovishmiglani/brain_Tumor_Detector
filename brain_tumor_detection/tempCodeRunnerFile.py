import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os

model_path = '/Users/lovishmiglani/Desktop/brain_tumor_detection/BrainTumorDetec.h5'

# Check if the file exists
if os.path.exists(model_path):
    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Error: Model file not found at {model_path}")
model = load_model(model_path)
# Load the trained model
# model_path = '/Users/lovishmiglani/Desktop/brain_tumor_detection/BrainTumorDetec.h5'
# model = load_model(model_path)
def make_prediction(img):
    # Assuming your model expects input images of shape (64, 64, 3)
    input_img = cv2.resize(img, (64, 64))
    input_img = np.expand_dims(input_img, axis=0)
    input_img = input_img / 255.0  # Normalize pixel values if needed

    # Use model.predict instead of model.predict_classes for probability output
    prediction = model.predict(input_img)

    # Assuming binary classification, convert to class label (0 or 1)
    pred_class = 1 if prediction > 0.5 else 0

    return pred_class


def show_result(img):
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    pred = make_prediction(img)
    if pred:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor")

def main():
    st.title("Brain Tumor Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        show_result(image)

if __name__ == '__main__':
    main()
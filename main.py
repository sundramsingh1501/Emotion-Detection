import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained models (only once)
@st.cache_resource
def load_emotion_model():
    face_classifier = cv2.CascadeClassifier(r'C:\Users\SHIVAM\Emotion_Detection\haarcascade\haarcascade_frontalface_default.xml')
    classifier = load_model(r'C:\Users\SHIVAM\Emotion_Detection\Model.h5')
    return face_classifier, classifier

face_classifier, classifier = load_emotion_model()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Streamlit app title
st.title("Real-Time Emotion Detection")

# Initialize session state for video capture control
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

# Start/Stop buttons
start_button = st.button('Start Emotion Detector')
stop_button = st.button('Stop Emotion Detector')

if start_button:
    st.session_state.run_detection = True

if stop_button:
    st.session_state.run_detection = False

FRAME_WINDOW = st.image([])

# Start video capture
cap = cv2.VideoCapture(0)

while st.session_state.run_detection:
    ret, frame = cap.read()
    if not ret:
        st.warning("No frames detected.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)

            # Add the emotion label to the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame (BGR) to RGB format for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

# Release video capture if stop button is clicked
if not st.session_state.run_detection:
    cap.release()
    st.write("Emotion Detector Stopped.")

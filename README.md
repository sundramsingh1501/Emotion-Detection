# Real-Time Emotion Detection

This project is a real-time emotion detection system that uses a webcam feed to detect human faces and classify their emotions. The system is built using **TensorFlow**, **Keras**, **OpenCV**, and **Streamlit** for web deployment.

## Project Structure

- `main.py`: The main Python file containing the logic for real-time emotion detection and the Streamlit app interface.
- `requirements.txt`: List of required libraries for the project.
- `Model.h5`: Pre-trained Convolutional Neural Network (CNN) model used for emotion detection.
- `haarcascade_frontalface_default.xml`: Pre-trained face detection classifier for identifying faces in video frames.
- `Data`: Directory that can be used for storing datasets if needed for future model training.

## Features

1. **Real-time Emotion Detection**: Uses a webcam feed to detect faces and predict emotions in real-time.
2. **Streamlit Interface**: Provides a simple web-based interface to start and stop the emotion detection system.
3. **Pre-trained Model**: Utilizes a pre-trained deep learning model for accurate emotion classification.
4. **Face Detection**: Leverages OpenCV's Haar Cascade classifier to detect human faces in video frames.
5. **Emotion Classification**: The system can classify the following emotions:
   - Angry
   - Disgust
   - Fear
   - Happy
   - Neutral
   - Sad
   - Surprise

## How it Works

1. **Face Detection**: The system uses OpenCV's Haar Cascade classifier to identify faces in the video stream.
2. **Emotion Prediction**: Once a face is detected, the model (loaded from `Model.h5`) processes the face and predicts the emotion.
3. **Web Interface**: The entire system is wrapped inside a Streamlit application, making it easy to interact with through a web interface.

## Setup Instructions

### Step 1: Install Dependencies

To run the project, ensure that you have all the required dependencies listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt

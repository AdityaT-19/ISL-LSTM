import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os
import time
from gtts import gTTS
import tempfile
import pygame
from threading import Thread
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Hand Sign Recognition", page_icon="👋", layout="wide")

# Initialize pygame for audio playback
pygame.mixer.init()

# Function to speak text using gTTS (Google Text-to-Speech)
def speak_text(text):
    try:
        # Create a temporary file for the speech audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts = gTTS(text=text, lang='en')
            temp_filename = f.name
            tts.save(temp_filename)
        
        # Play the audio in a non-blocking way
        def play_audio():
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
        
        # Start audio playback in a separate thread
        Thread(target=play_audio).start()
    except Exception as e:
        st.warning(f"Speech error: {e}")

# Initialize for tracking speech cooldown
if "last_speech_time" not in st.session_state:
    st.session_state.last_speech_time = 0

# Define the LSTM model class (same as in the notebook)
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Select output at final time step
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Function to extract keypoints using MediaPipe
def extract_keypoints(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints), results.multi_hand_landmarks
    return None, None


# Streamlit UI
st.title("Real-time Hand Sign Recognition")
st.markdown(
    "This application recognizes hand signs using a trained LSTM model and your webcam."
)

# Sidebar for configuration
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", True)
show_fps = st.sidebar.checkbox("Show FPS", True)
enable_speech = st.sidebar.checkbox("Enable Speech Output", True)
speech_cooldown = st.sidebar.slider("Speech Cooldown (seconds)", 1, 5, 2)

# Initialize session state if it doesn't exist
if "previous_threshold" not in st.session_state:
    st.session_state.previous_threshold = confidence_threshold

# Initialize camera in session state if needed
if "camera" not in st.session_state:
    st.session_state.camera = None

# Initialize for keeping track of last spoken prediction
if "last_spoken_prediction" not in st.session_state:
    st.session_state.last_spoken_prediction = ""

# Model loading section
@st.cache_resource
def load_model():
    # Define model parameters
    input_size = 63  # 21 landmarks * 3 coordinates (x, y, z)
    hidden_size = 64
    num_layers = 2
    num_classes = 3  # Update based on your model

    # Label mapping
    label_map = {0: "bye", 1: "hello", 2: "thankyou"}

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load model
    model = GestureLSTM(input_size, hidden_size, num_layers, num_classes)
    model_path = os.path.join("models", "gesture_model.pt")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device, label_map
    else:
        return None, device, label_map


# Load the model
model, device, label_map = load_model()

if model is None:
    st.error(
        "Model file not found. Please make sure the model is in the correct location."
    )
    st.stop()
else:
    st.sidebar.success("Model loaded successfully!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to initialize or reinitialize the hands object
@st.cache_resource(hash_funcs={float: lambda x: x})
def initialize_hands(threshold):
    # Close previous hands instance if it exists
    if "hands" in st.session_state and st.session_state.hands:
        st.session_state.hands.close()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=threshold,
    )
    st.session_state.hands = hands
    return hands


# Function to safely release camera
def release_camera():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None


# Function to safely initialize camera
def init_camera():
    # Release any existing camera first
    release_camera()

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None

        # Test if camera works by reading a frame
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None

        st.session_state.camera = cap
        return cap
    except Exception as e:
        st.error(f"Camera initialization error: {e}")
        return None


# Initialize or reinitialize the hands object if the threshold has changed
if st.session_state.previous_threshold != confidence_threshold:
    st.session_state.previous_threshold = confidence_threshold
    hands = initialize_hands(confidence_threshold)
    # Also reinitialize camera when threshold changes
    release_camera()
else:
    hands = initialize_hands(confidence_threshold)

# Create a placeholder for the webcam feed
video_placeholder = st.empty()
info_placeholder = st.empty()
status_placeholder = st.empty()

# Initialize sequence buffer
sequence = []
sequence_length = 30
prediction_text = "Waiting for hand gestures..."
confidence = 0.0

# Start the webcam
cap = init_camera()

if cap is None:
    st.error(
        """
    Failed to access the webcam. Please try:
    1. Refreshing the page
    2. Checking if another application is using the camera
    3. Verifying browser camera permissions
    4. Restarting your computer
    """
    )
    st.stop()

# Main application loop
try:
    prev_time = time.time()

    while cap.isOpened():
        # Check if configuration has changed during loop execution
        if st.session_state.previous_threshold != confidence_threshold:
            # If changed, safely release resources and break loop
            release_camera()
            break

        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            # Try to recover by reinitializing camera
            cap = init_camera()
            if cap is None:
                st.error("Camera connection lost and could not be restored.")
                break
            continue

        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Extract keypoints and draw landmarks
        keypoints_result, hand_landmarks = extract_keypoints(frame, hands)

        # Draw hand landmarks if enabled
        if hand_landmarks and show_landmarks:
            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4
                    ),
                    mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2
                    ),
                )

        # Process keypoints for prediction
        if keypoints_result is not None:
            sequence.append(keypoints_result)
            if len(sequence) > sequence_length:
                sequence.pop(0)

        # When we have enough frames, perform inference
        if len(sequence) == sequence_length:
            input_seq = np.array(sequence)
            input_tensor = (
                torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_idx = torch.argmax(output, dim=1).item()
                confidence = probabilities[predicted_idx].item()

                if confidence > confidence_threshold:
                    current_prediction = label_map[predicted_idx]
                    prediction_text = f"Gesture: {current_prediction}"
                    
                    # Speak the prediction if enabled and conditions are met
                    current_time = time.time()
                    time_since_last_speech = current_time - st.session_state.last_speech_time
                    
                    if (enable_speech and 
                        current_prediction != st.session_state.last_spoken_prediction and
                        time_since_last_speech > speech_cooldown):
                        
                        speak_text(f"{current_prediction}")
                        st.session_state.last_spoken_prediction = current_prediction
                        st.session_state.last_speech_time = current_time
                else:
                    prediction_text = "Confidence too low"
                    # Don't reset last_spoken_prediction here to avoid rapid toggling

        # Display prediction on frame
        cv2.putText(
            frame,
            f"{prediction_text} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show FPS if enabled
        if show_fps:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Convert to RGB for display in Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Display info about sequence progress
        info_placeholder.info(f"Collected frames: {len(sequence)}/{sequence_length}")

        # Add a small delay to reduce CPU usage
        time.sleep(0.01)

except Exception as e:
    st.error(f"Error: {e}")
finally:
    # Release resources safely
    release_camera()
    if "hands" in st.session_state and st.session_state.hands:
        st.session_state.hands.close()
    st.write("Camera stopped.")

# Add instructions at the bottom
st.markdown(
    """
### Instructions
1. Make sure your hand is clearly visible in the camera view
2. Perform one of the gestures: 'hello', 'bye', or 'thankyou'
3. Hold the gesture steady for a few seconds
4. The prediction will appear on the video feed

### About
This application uses a trained LSTM neural network to recognize hand signs in real-time.
"""
)

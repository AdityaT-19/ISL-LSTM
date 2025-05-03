import streamlit as st
import torch
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading
import pyttsx3
import tempfile
import pygame
import os
import shutil
from utils import ImprovedGestureLSTM # Assuming your model and feature extraction are here

# --- Constants ---
DEFAULT_MODEL_PATH = "models/ISL_holistic_best.pt"
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_THRESHOLD = 0.7
INPUT_SIZE = 354  # IMPORTANT: Must match your model's expected input size
HIDDEN_SIZE = 128 # Example, adjust if your model differs
NUM_LAYERS = 2    # Example, adjust if your model differs

# --- Page Configuration ---
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="👐",
    layout="wide",
)

# --- TTS Manager ---
class TTSManager:
    def __init__(self):
        self.tts_engine = None
        self.pygame_initialized = False
        self.temp_dir = tempfile.mkdtemp()
        self.last_spoken_word = None
        self._init_engine()

    def _init_engine(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", 150)
            self.tts_engine.setProperty("volume", 1.0)
            pygame.init()
            pygame.mixer.init()
            self.pygame_initialized = True
            print("TTS engine initialized.")
        except Exception as e:
            st.sidebar.warning(f"TTS/Pygame init failed: {e}. Audio disabled.")
            self.tts_engine = None
            self.pygame_initialized = False

    def speak(self, text):
        if not self.tts_engine or not self.pygame_initialized:
            return
        if text in ["Waiting...", ""] or text == self.last_spoken_word:
            return
        if pygame.mixer.music.get_busy(): # Don't interrupt current speech
            return

        try:
            temp_file = os.path.join(self.temp_dir, f"speech_{hash(text)}.wav")
            if not os.path.exists(temp_file):
                self.tts_engine.save_to_file(text, temp_file)
                self.tts_engine.runAndWait()
                time.sleep(0.05) # Allow file write

            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                self.last_spoken_word = text
                print(f"TTS: Spoke '{text}'") # Debugging
            else:
                print(f"TTS Warning: Failed to generate/find {temp_file}")
        except Exception as e:
            print(f"TTS Error in speak: {e}")
            if "mixer system not initialized" in str(e):
                self.pygame_initialized = False

    def cleanup(self):
        if self.pygame_initialized:
            pygame.mixer.quit()
            pygame.quit()
            self.pygame_initialized = False
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("TTS temp directory cleaned.")
        except Exception as e:
            print(f"Error cleaning TTS temp directory: {e}")

# --- Sign Prediction Backend ---
class SignPredictor:
    def __init__(self, model_path, sequence_length):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model & Labels
        self.model = None
        self.label2idx = None
        self.idx2label = None
        self.num_classes = 0

        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Processing State
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.previous_result = None # For potential velocity calculation

        # Threading Control
        self.is_running = False
        self.stop_event = threading.Event()
        self.processing_thread = None

        # Latest Results (for main thread access)
        self.latest_frame = None
        self.latest_prediction = "Stopped"
        self.latest_confidence = 0.0

        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.label2idx = checkpoint["label2idx"]
            self.idx2label = checkpoint["idx2label"]
            self.num_classes = len(self.idx2label)

            # Ensure these match your saved model's architecture
            self.model = ImprovedGestureLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=self.num_classes,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully ({self.num_classes} classes).")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None # Ensure model is None if loading fails

    def _extract_base_keypoints(self, results):
        """Extract base keypoints for velocity calculation (matches training)."""
        # Extract left hand landmarks (63 features)
        lh = np.zeros(63)
        if results.left_hand_landmarks:
             lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

        # Extract right hand landmarks (63 features)
        rh = np.zeros(63)
        if results.right_hand_landmarks:
             rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

        # Extract pose landmarks (upper body only - 9 landmarks * 3 coords = 27 features)
        upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        pose = np.zeros(len(upper_body_indices) * 3)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, idx in enumerate(upper_body_indices):
                if idx < len(landmarks):
                    pose[i*3:(i*3)+3] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

        # Calculate relative positions (9 features)
        relative_features = np.zeros(9)
        if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            landmarks = results.pose_landmarks.landmark # Re-access pose landmarks
            # Reference points (shoulders)
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z]) if 11 < len(landmarks) else np.zeros(3)
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z]) if 12 < len(landmarks) else np.zeros(3)

            # Hand center points relative to shoulders
            if results.left_hand_landmarks:
                left_hand_center = np.mean(np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]), axis=0)
                relative_features[0:3] = left_hand_center - left_shoulder
            # else: keep zeros

            if results.right_hand_landmarks:
                right_hand_center = np.mean(np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]), axis=0)
                relative_features[3:6] = right_hand_center - right_shoulder
            # else: keep zeros

            # Distance between hands (if both detected)
            if results.left_hand_landmarks and results.right_hand_landmarks:
                # Need centers calculated above
                if np.any(relative_features[0:3]) and np.any(relative_features[3:6]): # Check if centers were calculated
                     hand_distance = (relative_features[3:6] + right_shoulder) - (relative_features[0:3] + left_shoulder) # Recalculate absolute distance
                     relative_features[6:9] = hand_distance
            # else: keep zeros
        # else: keep zeros

        # Total base features = 63 + 63 + 27 + 9 = 162
        return np.concatenate([lh, rh, pose, relative_features])

    def _extract_hand_shape_features(self, results):
        """Extract features related to hand shape configuration (matches training)."""
        shape_features = [] # List to store features

        # For right hand (2 features)
        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            fingertips = [4, 8, 12, 16, 20] # Thumb, index, middle, ring, pinky
            fingertip_positions = [np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]) for idx in fingertips]

            finger_spread = 0
            for i in range(len(fingertips)-1):
                finger_spread += np.linalg.norm(fingertip_positions[i] - fingertip_positions[i+1])

            thumb_tip = fingertip_positions[0]
            index_tip = fingertip_positions[1]
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)

            shape_features.extend([finger_spread, pinch_distance])
        else:
            shape_features.extend([0.0, 0.0]) # Append zeros if no hand

        # For left hand (2 features)
        if results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark
            fingertips = [4, 8, 12, 16, 20]
            fingertip_positions = [np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]) for idx in fingertips]

            finger_spread = 0
            for i in range(len(fingertips)-1):
                finger_spread += np.linalg.norm(fingertip_positions[i] - fingertip_positions[i+1])

            thumb_tip = fingertip_positions[0]
            index_tip = fingertip_positions[1]
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)

            shape_features.extend([finger_spread, pinch_distance])
        else:
            shape_features.extend([0.0, 0.0]) # Append zeros if no hand

        # Total shape features = 4
        return np.array(shape_features)

    def _extract_keypoints(self, results):
        """
        Extract keypoints from MediaPipe results matching the training notebook's logic.
        """
        # Base features (lh, rh, pose, relative) = 162 features
        base_features = self._extract_base_keypoints(results)

        # Calculate velocity features (162 features)
        velocity_features = np.zeros_like(base_features)
        if self.previous_result is not None:
            previous_base_features = self._extract_base_keypoints(self.previous_result)
            # Ensure shapes match before subtraction
            if previous_base_features.shape == base_features.shape:
                velocity_features = base_features - previous_base_features

        # Extract hand shape dynamics (4 features)
        shape_features = self._extract_hand_shape_features(results)

        # Combine features: 162 (base) + 162 (velocity) + 4 (shape) = 328 features
        combined_features = np.concatenate([
            base_features,
            velocity_features,
            shape_features
        ])

        # Pad to exactly INPUT_SIZE (354)
        current_len = len(combined_features)
        if current_len < INPUT_SIZE:
            padding_size = INPUT_SIZE - current_len
            padding = np.zeros(padding_size)
            features = np.concatenate([combined_features, padding])
        elif current_len > INPUT_SIZE:
            # This shouldn't happen if calculations match training, but truncate as failsafe
            features = combined_features[:INPUT_SIZE]
            print(f"Warning: Truncated features from {current_len} to {INPUT_SIZE}")
        else:
            features = combined_features

        # Final check (optional but good for debugging)
        if features.shape[0] != INPUT_SIZE:
             raise ValueError(f"Final feature dimension mismatch: {features.shape[0]} != {INPUT_SIZE}")

        # Update previous result for next frame's velocity calculation
        self.previous_result = results

        return features

    def _draw_landmarks(self, frame, results):
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())
        self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())

    def _processing_loop(self):
        """Internal loop run in a background thread."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            self.is_running = False
            return

        while self.is_running and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame.")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            current_pred = "Waiting..."
            current_conf = 0.0

            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                self._draw_landmarks(frame_bgr, results)
                try:
                    features = self._extract_keypoints(results)
                    self.sequence_buffer.append(features)

                    if len(self.sequence_buffer) == self.sequence_length and self.model:
                        sequence_array = np.array(self.sequence_buffer)
                        sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            outputs, _ = self.model(sequence_tensor) # Assuming model returns (outputs, attention_weights)
                            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                            current_conf, predicted_idx_tensor = torch.max(probs, dim=0)
                            predicted_idx = predicted_idx_tensor.item()
                            current_conf = current_conf.item()
                            current_pred = self.idx2label.get(predicted_idx, "Unknown")
                    elif len(self.sequence_buffer) < self.sequence_length:
                         current_pred = f"Collecting... ({len(self.sequence_buffer)}/{self.sequence_length})"

                except Exception as e:
                    print(f"Error during processing: {e}")
                    current_pred = "Error"
                    current_conf = 0.0
                    self.sequence_buffer.clear() # Clear buffer on error
            else:
                # No landmarks detected, maybe clear buffer? Or keep prediction?
                if len(self.sequence_buffer) > 0:
                     self.sequence_buffer.clear() # Clear if no landmarks detected after having some
                current_pred = "No landmarks"
                current_conf = 0.0


            # Update shared state for the main thread
            self.latest_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.latest_prediction = current_pred
            self.latest_confidence = current_conf

            time.sleep(0.01) # Small sleep to prevent high CPU

        cap.release()
        print("Processing loop stopped.")

    def start(self):
        if not self.is_running:
            if not self.model:
                 st.error("Model not loaded. Cannot start.")
                 return
            self.is_running = True
            self.stop_event.clear()
            self.sequence_buffer.clear()
            self.latest_prediction = "Initializing..."
            self.latest_confidence = 0.0
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            print("Processing thread started.")
        else:
            print("Processing already running.")

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            self.latest_prediction = "Stopped"
            self.latest_confidence = 0.0
            self.latest_frame = None # Clear last frame
            print("Processing stopped.")

    def cleanup(self):
        self.stop()
        if self.holistic:
            try:
                self.holistic.close()
            except Exception as e:
                print(f"Error closing MediaPipe: {e}")

# --- Streamlit Main Application ---
def main():
    st.title("Live Sign Language Recognition")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, DEFAULT_THRESHOLD, 0.05)
    tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)

    st.sidebar.header("Controls")
    start_pressed = st.sidebar.button("Start Webcam", key="start")
    stop_pressed = st.sidebar.button("Stop Webcam", key="stop")

    # --- Initialize Backend Components in Session State ---
    if 'predictor' not in st.session_state:
        try:
            # Using default path and sequence length for simplicity now
            st.session_state.predictor = SignPredictor(DEFAULT_MODEL_PATH, DEFAULT_SEQUENCE_LENGTH)
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            st.stop() # Stop if predictor fails to init

    if 'tts_manager' not in st.session_state:
        st.session_state.tts_manager = TTSManager()

    predictor = st.session_state.predictor
    tts_manager = st.session_state.tts_manager

    # --- Control Logic ---
    if start_pressed and not predictor.is_running:
        predictor.start()
    if stop_pressed and predictor.is_running:
        predictor.stop()

    # --- Main Area Layout ---
    col1, col2 = st.columns([3, 1]) # Video column, Info column

    with col1:
        st.subheader("Live Feed")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("Prediction Info")
        pred_text_placeholder = st.empty()
        st.markdown("Confidence:")
        conf_bar_placeholder = st.progress(0.0)
        st.markdown("TTS Output:")
        tts_word_placeholder = st.empty()

    # --- UI Update Loop ---
    last_spoken_pred = None # Track what was last sent to TTS

    while True: # Streamlit reruns the script, this loop updates UI based on state
        if predictor.is_running:
            frame = predictor.latest_frame
            prediction = predictor.latest_prediction
            confidence = predictor.latest_confidence

            if frame is not None:
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            else:
                 with frame_placeholder.container():
                     st.info("Initializing feed...")

            pred_text_placeholder.markdown(f"### {prediction}")
            conf_bar_placeholder.progress(confidence)

            # TTS Logic
            tts_word_placeholder.write(f"Last Spoken: {tts_manager.last_spoken_word if tts_manager.last_spoken_word else '...'}")
            if tts_enabled and confidence >= confidence_threshold and prediction != last_spoken_pred:
                 # Check if prediction is a valid word (not status messages)
                 if prediction not in ["Waiting...", "Collecting...", "Initializing...", "Error", "No landmarks", "Unknown", "Stopped"] and not prediction.startswith("Collecting"):
                     tts_manager.speak(prediction)
                     last_spoken_pred = prediction # Update last spoken attempt

            # Check if stop was requested
            if not predictor.is_running:
                 break # Exit inner loop if stopped externally

            time.sleep(0.03) # Yield control

        else: # Not running state
            frame_placeholder.info("Press 'Start Webcam' in the sidebar.")
            pred_text_placeholder.markdown("### Stopped")
            conf_bar_placeholder.progress(0.0)
            tts_word_placeholder.write(f"Last Spoken: {tts_manager.last_spoken_word if tts_manager.last_spoken_word else '...'}")
            break # Exit loop when stopped

    # Note: Cleanup is tricky in standard Streamlit execution.
    # The TTS temp dir might not be cleaned until the Streamlit server stops.
    # Consider adding an explicit "Cleanup" button if needed.

if __name__ == "__main__":
    main()
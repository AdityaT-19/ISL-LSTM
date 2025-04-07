import streamlit as st
import torch
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading
from utils import ImprovedGestureLSTM

# Page configuration
st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="👐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Class for real-time sign language detection
class SignLanguageApp:
    def __init__(
        self,
        model_path="models/ISL_holistic_best.pt",
        sequence_length=30,
        threshold=0.7,
    ):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.load_model(model_path)

        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Initialize sequence parameters
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.sequence_buffer = []

        # For prediction smoothing and tracking
        self.prediction_history = deque(maxlen=5)
        self.current_prediction = None
        self.last_spoken_prediction = None

        # For feature calculation
        self.previous_result = None

        # For UI and visualization
        self.frame_placeholder = None
        self.status_placeholder = None
        self.confidence_bar = None
        self.prediction_text = None

        # For sentence building
        self.last_added_word = None
        self.last_added_time = 0

        # For detection control
        self.is_running = False
        self.stop_event = threading.Event()

    def load_model(self, model_path):
        """Load the trained model from checkpoint"""
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device)

            # Get label mapping
            self.label2idx = self.checkpoint["label2idx"]
            self.idx2label = self.checkpoint["idx2label"]

            # Create model
            input_size = 354  # Match your training feature dimension
            hidden_size = 128
            num_layers = 2
            num_classes = len(self.idx2label)

            self.model = ImprovedGestureLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
            )

            # Load model weights
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            st.sidebar.success(f"Model loaded successfully with {num_classes} classes.")

        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            st.stop()

    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        # Extract left hand landmarks
        lh = (
            np.zeros(63)
            if results.left_hand_landmarks is None
            else np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten()
        )

        # Extract right hand landmarks
        rh = (
            np.zeros(63)
            if results.right_hand_landmarks is None
            else np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten()
        )

        # Extract pose landmarks (upper body only)
        upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        pose = np.zeros(len(upper_body_indices) * 3)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, idx in enumerate(upper_body_indices):
                if idx < len(landmarks):
                    pose[i * 3 : (i * 3) + 3] = [
                        landmarks[idx].x,
                        landmarks[idx].y,
                        landmarks[idx].z,
                    ]

        # Calculate relative positions
        relative_features = []

        if results.pose_landmarks and (
            results.left_hand_landmarks or results.right_hand_landmarks
        ):
            # Reference points (shoulders)
            left_shoulder = (
                np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
                if 11 < len(landmarks)
                else np.zeros(3)
            )
            right_shoulder = (
                np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
                if 12 < len(landmarks)
                else np.zeros(3)
            )

            # Hand center points (if detected)
            if results.left_hand_landmarks:
                left_hand_center = np.mean(
                    np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in results.left_hand_landmarks.landmark
                        ]
                    ),
                    axis=0,
                )
                relative_features.extend(left_hand_center - left_shoulder)
            else:
                relative_features.extend(np.zeros(3))

            if results.right_hand_landmarks:
                right_hand_center = np.mean(
                    np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in results.right_hand_landmarks.landmark
                        ]
                    ),
                    axis=0,
                )
                relative_features.extend(right_hand_center - right_shoulder)
            else:
                relative_features.extend(np.zeros(3))

            # Distance between hands (if both detected)
            if results.left_hand_landmarks and results.right_hand_landmarks:
                hand_distance = right_hand_center - left_hand_center
                relative_features.extend(hand_distance)
            else:
                relative_features.extend(np.zeros(3))
        else:
            # Add placeholder zeros if landmarks aren't detected
            relative_features = np.zeros(9)

        # Base features
        base_features = np.concatenate([lh, rh, pose, relative_features])

        # Calculate velocity features if we have a previous frame
        if self.previous_result is not None:
            previous_features = self.extract_base_keypoints(self.previous_result)
            velocity_features = base_features - previous_features
        else:
            velocity_features = np.zeros_like(base_features)

        # Extract hand shape dynamics
        shape_features = self.extract_hand_shape_features(results)

        # Additional features to match dimensions
        additional_features = np.zeros(
            354 - len(base_features) - len(velocity_features) - len(shape_features)
        )

        # Combine all features
        features = np.concatenate(
            [base_features, velocity_features, shape_features, additional_features]
        )

        # Update previous result for next frame
        self.previous_result = results

        # Ensure exactly 354 dimensions
        assert features.shape[0] == 354, (
            f"Feature dimension mismatch: {features.shape[0]} != 354"
        )

        return features

    def extract_base_keypoints(self, results):
        """Extract base keypoints for velocity calculation"""
        # Extract left hand landmarks
        lh = (
            np.zeros(63)
            if results.left_hand_landmarks is None
            else np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten()
        )

        # Extract right hand landmarks
        rh = (
            np.zeros(63)
            if results.right_hand_landmarks is None
            else np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten()
        )

        # Extract pose landmarks
        upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        pose = np.zeros(len(upper_body_indices) * 3)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, idx in enumerate(upper_body_indices):
                if idx < len(landmarks):
                    pose[i * 3 : (i * 3) + 3] = [
                        landmarks[idx].x,
                        landmarks[idx].y,
                        landmarks[idx].z,
                    ]

        # Calculate relative positions
        relative_features = []

        if results.pose_landmarks and (
            results.left_hand_landmarks or results.right_hand_landmarks
        ):
            left_shoulder = (
                np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
                if 11 < len(landmarks)
                else np.zeros(3)
            )
            right_shoulder = (
                np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
                if 12 < len(landmarks)
                else np.zeros(3)
            )

            if results.left_hand_landmarks:
                left_hand_center = np.mean(
                    np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in results.left_hand_landmarks.landmark
                        ]
                    ),
                    axis=0,
                )
                relative_features.extend(left_hand_center - left_shoulder)
            else:
                relative_features.extend(np.zeros(3))

            if results.right_hand_landmarks:
                right_hand_center = np.mean(
                    np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in results.right_hand_landmarks.landmark
                        ]
                    ),
                    axis=0,
                )
                relative_features.extend(right_hand_center - right_shoulder)
            else:
                relative_features.extend(np.zeros(3))

            if results.left_hand_landmarks and results.right_hand_landmarks:
                hand_distance = right_hand_center - left_hand_center
                relative_features.extend(hand_distance)
            else:
                relative_features.extend(np.zeros(3))
        else:
            relative_features = np.zeros(9)

        return np.concatenate([lh, rh, pose, relative_features])

    def extract_hand_shape_features(self, results):
        """Extract features related to hand shape configuration"""
        shape_features = []

        # For right hand
        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark

            # Calculate finger spread
            fingertips = [4, 8, 12, 16, 20]
            fingertip_positions = [
                np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
                for idx in fingertips
            ]

            finger_spread = 0
            for i in range(len(fingertips) - 1):
                finger_spread += np.linalg.norm(
                    fingertip_positions[i] - fingertip_positions[i + 1]
                )

            # Thumb-index pinch distance
            thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
            index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)

            shape_features.extend([finger_spread, pinch_distance])
        else:
            shape_features.extend([0, 0])

        # For left hand
        if results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark

            fingertips = [4, 8, 12, 16, 20]
            fingertip_positions = [
                np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
                for idx in fingertips
            ]

            finger_spread = 0
            for i in range(len(fingertips) - 1):
                finger_spread += np.linalg.norm(
                    fingertip_positions[i] - fingertip_positions[i + 1]
                )

            thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
            index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)

            shape_features.extend([finger_spread, pinch_distance])
        else:
            shape_features.extend([0, 0])

        return np.array(shape_features)

    def draw_landmarks(self, frame, results):
        """Draw MediaPipe landmarks on the frame"""
        # Draw pose connections
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        # Draw hand connections
        self.mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
        )

        self.mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
        )

    def smooth_prediction(self, prediction_idx, confidence):
        """Apply temporal smoothing to predictions"""
        if confidence > self.threshold:
            self.prediction_history.append(prediction_idx)

            # Return the most common prediction from the recent history
            if len(self.prediction_history) > 2:
                # Count occurrences of each prediction
                prediction_counts = {}
                for pred in self.prediction_history:
                    if pred in prediction_counts:
                        prediction_counts[pred] += 1
                    else:
                        prediction_counts[pred] = 1

                # Find the most common prediction
                most_common_pred = max(prediction_counts, key=prediction_counts.get)
                return most_common_pred, self.idx2label[most_common_pred]

        # If confidence is too low or not enough history
        if len(self.prediction_history) > 0:
            last_pred = self.prediction_history[-1]
            return last_pred, self.idx2label[last_pred]

        # Default if no predictions yet
        return -1, "Waiting..."

    def set_current_word(self, word):
        """Add recognized word to the current sentence"""

        # Only add if it's a new word and not "Waiting..." and enough time has passed
        if (
            word != self.last_added_word
            and word != "Waiting..."
            and word != "Low confidence"
        ):
            self.current_word = word

            # Update the sentence display
            if self.sentence_placeholder:
                self.sentence_placeholder.markdown(
                    f"**Current Word:** {self.current_word}"
                )
        else:
            self.current_word = f"Waiting... -- {word}"

    def process_frame(self, frame):
        """Process a single frame for sign language detection"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)

        # Draw landmarks if detected
        if (
            results.pose_landmarks
            or results.left_hand_landmarks
            or results.right_hand_landmarks
        ):
            self.draw_landmarks(frame, results)

            # Extract features
            features = self.extract_keypoints(results)

            # Add to sequence buffer
            self.sequence_buffer.append(features)
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)

        # Update buffer fullness indicator
        buffer_fullness = len(self.sequence_buffer) / self.sequence_length
        if self.buffer_indicator:
            self.buffer_indicator.progress(buffer_fullness)

        # Make prediction when buffer is full
        if len(self.sequence_buffer) == self.sequence_length:
            # Convert buffer to tensor
            sequence = np.array(self.sequence_buffer)
            sequence_tensor = (
                torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

            # Get model prediction
            with torch.no_grad():
                outputs, attention_weights = self.model(sequence_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

                # Get highest probability prediction
                predicted_idx = torch.argmax(probs).item()
                confidence = probs[predicted_idx].item()

                # Apply smoothing
                final_idx, prediction = self.smooth_prediction(
                    predicted_idx, confidence
                )

                # Update UI with prediction
                if self.prediction_text:
                    self.prediction_text.markdown(f"## Prediction: **{prediction}**")

                if self.confidence_bar:
                    self.confidence_bar.progress(confidence)

                # Add to sentence if confidence is high enough
                if confidence > self.threshold:
                    self.set_current_word(prediction)

                self.current_prediction = prediction

        # Add text overlay to frame
        buffer_text = f"Frames: {len(self.sequence_buffer)}/{self.sequence_length}"
        cv2.putText(
            frame,
            buffer_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame

    def start_webcam(self):
        """Start the webcam capture and processing loop"""
        self.is_running = True
        self.stop_event.clear()

        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam!")
            return

        # Create UI placeholders for updates
        frame_col, info_col = st.columns([2, 1])

        with frame_col:
            self.frame_placeholder = st.empty()
            self.buffer_indicator = st.progress(0)

        with info_col:
            self.prediction_text = st.empty()
            st.markdown("### Confidence")
            self.confidence_bar = st.progress(0)
            st.markdown("### Current Word")
            self.sentence_placeholder = st.empty()
            self.sentence_placeholder.markdown("**Current Word:** ")

        # Main webcam loop
        while cap.isOpened() and self.is_running and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam!")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Process the frame
            processed_frame = self.process_frame(frame)

            # Convert to RGB for display in Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Update the frame placeholder
            self.frame_placeholder.image(
                processed_frame_rgb, channels="RGB", use_container_width=True
            )

            # Sleep to avoid overwhelming CPU
            time.sleep(0.01)

        # Release resources
        cap.release()
        self.holistic.close()
        self.is_running = False

    def stop_webcam(self):
        """Stop the webcam capture"""
        self.stop_event.set()
        self.is_running = False


def main():
    # Set up the sidebar
    st.sidebar.title("Indian Sign Language Recognition")

    # Application modes - simplified
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Live Recognition", "About"]
    )

    # Initialize the sign language app
    sign_app = SignLanguageApp()

    if app_mode == "Live Recognition":
        st.markdown("# Live Sign Language Recognition")

        # Start/stop webcam buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Webcam")
        with col2:
            stop_button = st.button("Stop Webcam")

        if start_button:
            sign_app.start_webcam()

        if stop_button:
            sign_app.stop_webcam()

    elif app_mode == "About":
        st.markdown("# About This Project")
        st.markdown("""
        ## Indian Sign Language Recognition System
        
        This application uses computer vision and deep learning to recognize Indian Sign Language gestures in real-time.
        
        ### Technology Stack:
        - **Frontend**: Streamlit
        - **Computer Vision**: MediaPipe for hand and pose tracking
        - **Machine Learning**: PyTorch LSTM with attention mechanism
        - **Text-to-Speech**: pyttsx3
        """)


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
from collections import deque
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    

class AttentionModule(nn.Module):
    """Self-attention module for temporal feature refinement"""
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Apply softmax over sequence dimension
        
        # Apply attention weights
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, attn_weights

class ImprovedGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5, bidirectional=True):
        """
        Enhanced LSTM model with attention and residual connections.
        
        Args:
            input_size: Number of features per frame
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(ImprovedGestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.direction_factor = 2 if bidirectional else 1
        
        # Feature embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # Main LSTM layers
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism 
        self.attention = AttentionModule(hidden_size * self.direction_factor)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * self.direction_factor, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Embed features
        x = self.embedding(x)  # (batch, seq_len, hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*direction_factor)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classification layers with residual connection
        out = self.fc1(context)
        out = self.layernorm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights
    

class RealTimeSignLanguageDetection:
    def __init__(self, model_path="../models/ISL_holistic_best.pt", sequence_length=30, threshold=0.7):
        """
        Initialize the real-time sign language detection system.
        
        Args:
            model_path: Path to the saved model checkpoint
            sequence_length: Length of the sequence to be fed to the model
            threshold: Confidence threshold for predictions
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model(model_path)
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize sequence parameters
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.sequence_buffer = []
        
        # For prediction smoothing
        self.prediction_history = deque(maxlen=5)
        
        # For feature calculation
        self.previous_result = None
        self.previous_results_buffer = []
        
        # For FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
    def load_model(self, model_path):
        """Load the trained model from checkpoint"""
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get label mapping
            self.label2idx = self.checkpoint['label2idx']
            self.idx2label = self.checkpoint['idx2label']
            
            # Create model
            input_size = 354  # Match your training feature dimension
            hidden_size = 128
            num_layers = 2
            num_classes = len(self.idx2label)
            
            self.model = ImprovedGestureLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes
            )
            
            # Load model weights
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded with {num_classes} classes: {list(self.idx2label.values())}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def extract_keypoints(self, results):
        """
        Extract keypoints from MediaPipe results in the same format used during training.
        """
        # Extract left hand landmarks
        lh = np.zeros(63) if results.left_hand_landmarks is None else \
             np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        
        # Extract right hand landmarks
        rh = np.zeros(63) if results.right_hand_landmarks is None else \
             np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        
        # Extract pose landmarks (upper body only)
        upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        pose = np.zeros(len(upper_body_indices) * 3)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, idx in enumerate(upper_body_indices):
                if idx < len(landmarks):
                    pose[i*3:(i*3)+3] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
        
        # Calculate relative positions
        relative_features = []
        
        if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            # Reference points (shoulders)
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z]) if 11 < len(landmarks) else np.zeros(3)
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z]) if 12 < len(landmarks) else np.zeros(3)
            
            # Hand center points (if detected)
            if results.left_hand_landmarks:
                left_hand_center = np.mean(np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]), axis=0)
                relative_features.extend(left_hand_center - left_shoulder)
            else:
                relative_features.extend(np.zeros(3))
            
            if results.right_hand_landmarks:
                right_hand_center = np.mean(np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]), axis=0)
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
        
        # Additional features used in training - placeholders to match dimensions
        additional_features = np.zeros(354 - len(base_features) - len(velocity_features) - len(shape_features))
        
        # Combine all features
        features = np.concatenate([
            base_features,
            velocity_features,
            shape_features,
            additional_features
        ])
        
        # Update previous result for next frame
        self.previous_result = results
        
        # Ensure exactly 354 dimensions
        assert features.shape[0] == 354, f"Feature dimension mismatch: {features.shape[0]} != 354"
        
        return features
    
    def extract_base_keypoints(self, results):
        """Extract base keypoints for velocity calculation"""
        # Extract left hand landmarks
        lh = np.zeros(63) if results.left_hand_landmarks is None else \
             np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        
        # Extract right hand landmarks
        rh = np.zeros(63) if results.right_hand_landmarks is None else \
             np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        
        # Extract pose landmarks
        upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        pose = np.zeros(len(upper_body_indices) * 3)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, idx in enumerate(upper_body_indices):
                if idx < len(landmarks):
                    pose[i*3:(i*3)+3] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
        
        # Calculate relative positions
        relative_features = []
        
        if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            # Reference points (shoulders)
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z]) if 11 < len(landmarks) else np.zeros(3)
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z]) if 12 < len(landmarks) else np.zeros(3)
            
            # Hand center points (if detected)
            if results.left_hand_landmarks:
                left_hand_center = np.mean(np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]), axis=0)
                relative_features.extend(left_hand_center - left_shoulder)
            else:
                relative_features.extend(np.zeros(3))
            
            if results.right_hand_landmarks:
                right_hand_center = np.mean(np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]), axis=0)
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
        
        return np.concatenate([lh, rh, pose, relative_features])
    
    def extract_hand_shape_features(self, results):
        """Extract features related to hand shape configuration"""
        shape_features = []
        
        # For right hand
        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            
            # Calculate finger spread (distance between fingertips)
            fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
            fingertip_positions = [np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]) 
                                  for idx in fingertips]
            
            # Measure spread: sum of distances between adjacent fingertips
            finger_spread = 0
            for i in range(len(fingertips)-1):
                finger_spread += np.linalg.norm(fingertip_positions[i] - fingertip_positions[i+1])
            
            # Thumb-index pinch distance (important for many signs)
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
            fingertip_positions = [np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]) 
                                  for idx in fingertips]
            
            finger_spread = 0
            for i in range(len(fingertips)-1):
                finger_spread += np.linalg.norm(fingertip_positions[i] - fingertip_positions[i+1])
            
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
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Draw hand connections
        self.mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())
        
        self.mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())

    def calculate_fps(self):
        """Calculate and update frames per second"""
        self.fps_counter += 1
        current_time = time.time()
        if (current_time - self.fps_start_time) >= 1:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def smooth_prediction(self, prediction_idx, confidence):
        """Apply temporal smoothing to predictions to reduce flickering"""
        if confidence > self.threshold:
            self.prediction_history.append(prediction_idx)
            
            # Return the most common prediction from the recent history
            if len(self.prediction_history) > 2:
                # Fixed: Handle mode function properly for different scipy versions
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
    
    def process_frame(self, frame):
        """Process a single frame for sign language recognition"""
        self.calculate_fps()
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)
        
        # Initialize prediction variables
        prediction = "Collecting frames..."
        confidence = 0.0
        final_idx = -1
        attention_weights = None
        
        # Draw landmarks if detected
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            self.draw_landmarks(frame, results)
            
            # Extract features
            features = self.extract_keypoints(results)
            
            # Add to sequence buffer
            self.sequence_buffer.append(features)
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)
            
            # Make prediction when buffer is full and model is loaded
            if len(self.sequence_buffer) == self.sequence_length and self.model is not None:
                # Convert buffer to tensor
                sequence = np.array(self.sequence_buffer)
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    outputs, attn_weights = self.model(sequence_tensor)
                    probs = F.softmax(outputs, dim=1)[0]
                    
                    # Get highest probability prediction
                    predicted_idx = torch.argmax(probs).item()
                    confidence = probs[predicted_idx].item()
                    
                    # Apply smoothing
                    final_idx, prediction = self.smooth_prediction(predicted_idx, confidence)
                    attention_weights = attn_weights.squeeze().cpu().numpy()
        
        # Return results
        buffer_fullness = len(self.sequence_buffer) / self.sequence_length
        return frame, prediction, confidence, buffer_fullness, self.fps, final_idx, attention_weights

    
    def start_detection(self):
        """Start real-time sign language detection using webcam"""
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break
            
            # Calculate FPS
            self.calculate_fps()
            
            # Mirror display (more intuitive for user)
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(frame_rgb)
            
            # Draw landmarks if detected
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                self.draw_landmarks(frame, results)
                
                # Extract features
                features = self.extract_keypoints(results)
                
                # Add to sequence buffer
                self.sequence_buffer.append(features)
                if len(self.sequence_buffer) > self.sequence_length:
                    self.sequence_buffer.pop(0)
            
            # Initialize prediction text
            prediction = "Collecting frames..."
            confidence = 0.0
            
            # Make prediction when buffer is full
            if len(self.sequence_buffer) == self.sequence_length:
                # Convert buffer to tensor
                sequence = np.array(self.sequence_buffer)
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    outputs, attention_weights = self.model(sequence_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Get highest probability prediction
                    predicted_idx = torch.argmax(probs).item()
                    confidence = probs[predicted_idx].item()
                    
                    # Apply smoothing
                    final_idx, prediction = self.smooth_prediction(predicted_idx, confidence)
            
            # Create overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 130), (0, 0, 0), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add text information
            cv2.putText(frame, f"Prediction: {prediction}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS: {self.fps}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            
            # Display sequence buffer status bar
            buffer_width = frame.shape[1] - 40
            buffer_height = 15
            buffer_fullness = len(self.sequence_buffer) / self.sequence_length
            cv2.rectangle(frame, (20, 150), (20 + buffer_width, 150 + buffer_height), (100, 100, 100), 2)
            cv2.rectangle(frame, (20, 150), (20 + int(buffer_width * buffer_fullness), 150 + buffer_height), 
                         (0, 255, 0), -1)
            
            # Display frame
            cv2.imshow("Sign Language Recognition", frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()

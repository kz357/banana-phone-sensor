import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import time
import random
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Banana Phone Detector 🍌📞",
    page_icon="🍌",
    layout="wide"
)

# Initialize session state variables
if 'sound_playing' not in st.session_state:
    st.session_state.sound_playing = False
if 'banana_near_face_start_time' not in st.session_state:
    st.session_state.banana_near_face_start_time = None

# Initialize MediaPipe solutions
@st.cache_resource
def load_mediapipe():
    mp_face_detection = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return mp_face_detection, mp_hands, mp_drawing, face_detection, hands

# Initialize YOLO model
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

# Load models
mp_face_detection, mp_hands, mp_drawing, face_detection, hands = load_mediapipe()
yolo_model = load_yolo()

# Audio handling for Streamlit
def create_audio_player(play_sound=False):
    """Create an HTML audio player that can be controlled"""
    if play_sound:
        audio_html = """
        <audio id="banana_sound" autoplay loop>
            <source src="data:audio/mp3;base64,{}" type="audio/mp3">
        </audio>
        <script>
            var audio = document.getElementById('banana_sound');
            audio.play();
        </script>
        """
        try:
            # Try to read and encode the audio file
            with open('soundbyte.mp3', 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                return audio_html.format(audio_b64)
        except:
            return "<p>Could not load audio file</p>"
    else:
        return """
        <script>
            var audio = document.getElementById('banana_sound');
            if (audio) {
                audio.pause();
            }
        </script>
        """

def is_banana_near_face_like_phone(banana_box, face_box, hand_landmarks_list, image_shape):
    """Check if banana is positioned like a phone near face"""
    if banana_box is None or face_box is None or not hand_landmarks_list:
        return False
    
    # Get banana center and dimensions
    banana_x1, banana_y1, banana_x2, banana_y2 = banana_box
    banana_center_x = (banana_x1 + banana_x2) / 2
    banana_center_y = (banana_y1 + banana_y2) / 2
    banana_width = banana_x2 - banana_x1
    banana_height = banana_y2 - banana_y1
    
    # Get face center and dimensions
    face_x1, face_y1, face_x2, face_y2 = face_box
    face_center_x = (face_x1 + face_x2) / 2
    face_center_y = (face_y1 + face_y2) / 2
    face_width = face_x2 - face_x1
    
    # Check if banana is roughly vertical (height > width)
    is_vertical = banana_height > banana_width * 0.8
    
    # Check if banana is close to face (within reasonable distance)
    distance_x = abs(banana_center_x - face_center_x)
    distance_y = abs(banana_center_y - face_center_y)
    max_distance = face_width * 1.5
    
    is_near_face = distance_x < max_distance and distance_y < max_distance
    
    # Check if hand is near banana
    h, w = image_shape[:2]
    hand_near_banana = False
    
    for hand_landmarks in hand_landmarks_list:
        wrist = hand_landmarks.landmark[0]
        hand_x = wrist.x * w
        hand_y = wrist.y * h
        
        hand_banana_distance = np.sqrt((hand_x - banana_center_x)**2 + (hand_y - banana_center_y)**2)
        if hand_banana_distance < max_distance:
            hand_near_banana = True
            break
    
    return is_vertical and is_near_face and hand_near_banana

def draw_yello_text(image, width, height):
    """Draw 'yello?' text repeatedly across the screen"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (0, 255, 255)  # Yellow color
    
    text = "yello?"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate spacing
    x_spacing = text_size[0] + 20
    y_spacing = text_size[1] + 15
    
    # Draw text in a grid pattern
    for y in range(text_size[1], height, y_spacing):
        for x in range(0, width, x_spacing):
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-3, 3)
            cv2.putText(image, text, (x + offset_x, y + offset_y), 
                       font, font_scale, color, thickness)

def process_frame(frame):
    """Process a single frame"""
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Process MediaPipe detections
    face_results = face_detection.process(image_rgb)
    hand_results = hands.process(image_rgb)
    
    # Convert back to BGR for processing
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Run YOLO detection for bananas only
    yolo_results = yolo_model(image_bgr, classes=[46], conf=0.3, verbose=False)
    
    # Store detection data
    banana_box = None
    face_box = None
    hand_landmarks_list = []
    
    # Process face detection
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image_bgr.shape
            face_x1 = int(bbox.xmin * w)
            face_y1 = int(bbox.ymin * h)
            face_x2 = int((bbox.xmin + bbox.width) * w)
            face_y2 = int((bbox.ymin + bbox.height) * h)
            face_box = (face_x1, face_y1, face_x2, face_y2)
            
            mp_drawing.draw_detection(
                image_bgr, 
                detection,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    
    # Process hand detection
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_landmarks_list.append(hand_landmarks)
            
            h, w, _ = image_bgr.shape
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )
    
    # Process banana detection
    for result in yolo_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                banana_box = (x1, y1, x2, y2)
                
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                label = f'Banana: {confidence:.2f}'
                cv2.putText(image_bgr, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Check if banana is positioned like a phone near face
    is_phone_like = is_banana_near_face_like_phone(banana_box, face_box, hand_landmarks_list, image_bgr.shape)
    
    # Handle timing logic for sound delay
    current_time = time.time()
    should_play_sound = False
    
    if is_phone_like:
        if st.session_state.banana_near_face_start_time is None:
            st.session_state.banana_near_face_start_time = current_time
        
        time_elapsed = current_time - st.session_state.banana_near_face_start_time
        if time_elapsed >= 0.5:  # 0.5 second delay
            should_play_sound = True
            h, w = image_bgr.shape[:2]
            draw_yello_text(image_bgr, w, h)
    else:
        st.session_state.banana_near_face_start_time = None
        should_play_sound = False
        cv2.putText(image_bgr, "No banana phone :(", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    
    # Update sound playing state
    st.session_state.sound_playing = should_play_sound
    
    return image_bgr, should_play_sound

# Streamlit UI
st.title("🍌 Banana Phone Detector 📞")
st.write("Hold a banana near your face like a phone to hear the sound!")

# Create columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    # Video display
    video_placeholder = st.empty()

with col2:
    st.write("### Instructions:")
    st.write("1. Hold a banana vertically")
    st.write("2. Bring it close to your face")
    st.write("3. Wait 0.5 seconds")
    st.write("4. Listen for the sound!")
    
    # Status display
    status_placeholder = st.empty()

# Audio placeholder
audio_placeholder = st.empty()

# Camera setup
run = st.checkbox('Run Camera')

if run:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open camera. Please check your camera permissions.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Process frame
            processed_frame, should_play_sound = process_frame(frame)
            
            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            # Update status
            if should_play_sound:
                status_placeholder.success("🍌 Banana Phone Active! 📞")
                audio_placeholder.markdown(create_audio_player(True), unsafe_allow_html=True)
            else:
                status_placeholder.info("📱 No banana phone detected")
                audio_placeholder.markdown(create_audio_player(False), unsafe_allow_html=True)
            
            # Small delay to prevent overwhelming the browser
            time.sleep(0.1)
        
        cap.release()
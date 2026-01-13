import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import time

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load the sound file
try:
    sound = pygame.mixer.Sound('soundbyte.mp3')
    print("Sound loaded successfully!")
except pygame.error as e:
    print(f"Could not load sound file: {e}")
    sound = None

# Sound control variables
sound_playing = False
sound_thread = None

# Timing variables for delay
banana_near_face_start_time = None
banana_near_face_duration_threshold = 0.5  # 0.5 seconds

def play_sound_loop():
    """Play sound in a loop while sound_playing is True"""
    global sound_playing
    while sound_playing and sound is not None:
        sound.play()
        # Wait for sound to finish before looping
        while pygame.mixer.get_busy() and sound_playing:
            time.sleep(0.1)

def start_sound():
    """Start playing sound in loop"""
    global sound_playing, sound_thread
    if not sound_playing and sound is not None:
        sound_playing = True
        sound_thread = threading.Thread(target=play_sound_loop)
        sound_thread.daemon = True
        sound_thread.start()

def stop_sound():
    """Stop playing sound"""
    global sound_playing
    if sound_playing:
        sound_playing = False
        pygame.mixer.stop()

# Create MediaPipe detection objects
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

def is_banana_near_face_like_phone(banana_box, face_box, hand_landmarks_list, image_shape):
    """
    Check if banana is positioned like a phone near face
    """
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
    max_distance = face_width * 1.5  # Allow some distance from face
    
    is_near_face = distance_x < max_distance and distance_y < max_distance
    
    # Check if hand is near banana
    h, w = image_shape[:2]
    hand_near_banana = False
    
    for hand_landmarks in hand_landmarks_list:
        # Get hand center (using wrist landmark)
        wrist = hand_landmarks.landmark[0]  # Wrist is landmark 0
        hand_x = wrist.x * w
        hand_y = wrist.y * h
        
        # Check if hand is close to banana
        hand_banana_distance = np.sqrt((hand_x - banana_center_x)**2 + (hand_y - banana_center_y)**2)
        if hand_banana_distance < max_distance:
            hand_near_banana = True
            break
    
    return is_vertical and is_near_face and hand_near_banana

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Process MediaPipe detections
    face_results = face_detection.process(image_rgb)
    hand_results = hands.process(image_rgb)
    
    # Convert back to BGR for OpenCV and YOLO
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Run YOLO detection for bananas only (class ID 46)
    yolo_results = yolo_model(image_bgr, classes=[46], conf=0.3, verbose=False)
    
    # Store detection data for analysis
    banana_box = None
    face_box = None
    hand_landmarks_list = []
    
    # Draw face detection and store face box
    if face_results.detections:
        for detection in face_results.detections:
            # Get face bounding box
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
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Green box
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Blue landmarks
            )
    
    # Draw hand detection and store hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_landmarks_list.append(hand_landmarks)
            
            # Get bounding box for hand
            h, w, _ = image_bgr.shape
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding to bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Draw red bounding box around hand
            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # Draw hand landmarks and connections in magenta
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),  # Magenta landmarks
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)  # Magenta connections
            )
    
    # Draw banana detection and store banana box
    for result in yolo_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                banana_box = (x1, y1, x2, y2)
                
                # Draw yellow bounding box for banana
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                
                # Add confidence label
                label = f'Banana: {confidence:.2f}'
                cv2.putText(image_bgr, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Check if banana is positioned like a phone near face
    is_phone_like = is_banana_near_face_like_phone(banana_box, face_box, hand_landmarks_list, image_bgr.shape)
    
    # Handle timing logic for sound delay
    current_time = time.time()
    
    if is_phone_like:
        # Start timer if this is the first detection
        if banana_near_face_start_time is None:
            banana_near_face_start_time = current_time
        
        # Check if enough time has passed to start sound
        time_elapsed = current_time - banana_near_face_start_time
        if time_elapsed >= banana_near_face_duration_threshold:
            start_sound()
    else:
        # Reset timer and stop sound when banana is not near face
        banana_near_face_start_time = None
        stop_sound()
    
    # Display phone-like position indicator in top left
    status_text = f"Banana near face: {'true' if is_phone_like else 'false'}"
    text_color = (0, 255, 0) if is_phone_like else (0, 0, 255)  # Green if true, red if false
    
    cv2.putText(image_bgr, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Add sound status indicator
    sound_status = "Sound: Playing" if sound_playing else "Sound: Stopped"
    cv2.putText(image_bgr, sound_status, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display the image
    cv2.imshow('Face, Hand, and Banana Detection', image_bgr)
    
    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
stop_sound()
cap.release()
cv2.destroyAllWindows()
face_detection.close()
hands.close()
pygame.mixer.quit()
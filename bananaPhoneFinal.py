import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import time
import random

# initialize mp
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# yolo
yolo_model = YOLO('yolov8n.pt')

# for audio
pygame.mixer.init()

# load sound
try:
    sound = pygame.mixer.Sound('soundbyte.mp3')
    print("Sound loaded successfully!")
except pygame.error as e:
    print(f"Could not load sound file: {e}")
    sound = None

sound_playing = False
sound_thread = None

banana_near_face_start_time = None
banana_near_face_duration_threshold = 0.5  # 0.5 seconds

def play_sound_loop():
    """Play sound in a loop while sound_playing is True"""
    global sound_playing
    while sound_playing and sound is not None:
        sound.play()
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
    
    banana_x1, banana_y1, banana_x2, banana_y2 = banana_box
    banana_center_x = (banana_x1 + banana_x2) / 2
    banana_center_y = (banana_y1 + banana_y2) / 2
    banana_width = banana_x2 - banana_x1
    banana_height = banana_y2 - banana_y1
    
    face_x1, face_y1, face_x2, face_y2 = face_box
    face_center_x = (face_x1 + face_x2) / 2
    face_center_y = (face_y1 + face_y2) / 2
    face_width = face_x2 - face_x1
    
    is_vertical = banana_height > banana_width * 0.8
    
 
    distance_x = abs(banana_center_x - face_center_x)
    distance_y = abs(banana_center_y - face_center_y)
    max_distance = face_width * 1.5  
    
    is_near_face = distance_x < max_distance and distance_y < max_distance
    
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
    color = (0, 255, 255)  
    
    text = "yello?"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    x_spacing = text_size[0] + 20
    y_spacing = text_size[1] + 15
    
    for y in range(text_size[1], height, y_spacing):
        for x in range(0, width, x_spacing):
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-3, 3)
            cv2.putText(image, text, (x + offset_x, y + offset_y), 
                       font, font_scale, color, thickness)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    face_results = face_detection.process(image_rgb)
    hand_results = hands.process(image_rgb)
    
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    yolo_results = yolo_model(image_bgr, classes=[46], conf=0.3, verbose=False)
    
    banana_box = None
    face_box = None
    hand_landmarks_list = []
    
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
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Green box
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Blue landmarks
            )
    
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
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),  # Magenta landmarks
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)  # Magenta connections
            )
    
    for result in yolo_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                banana_box = (x1, y1, x2, y2)
                
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                
                label = f'Banana: {confidence:.2f}'
                cv2.putText(image_bgr, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    is_phone_like = is_banana_near_face_like_phone(banana_box, face_box, hand_landmarks_list, image_bgr.shape)
    
    current_time = time.time()
    
    if is_phone_like:

        if banana_near_face_start_time is None:
            banana_near_face_start_time = current_time
        

        time_elapsed = current_time - banana_near_face_start_time
        if time_elapsed >= banana_near_face_duration_threshold:
            start_sound()
           
            h, w = image_bgr.shape[:2]
            draw_yello_text(image_bgr, w, h)
    else:

        banana_near_face_start_time = None
        stop_sound()
        
       
        cv2.putText(image_bgr, "No banana phone :(", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    
    cv2.imshow('Face, Hand, and Banana Detection', image_bgr)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
stop_sound()
cap.release()
cv2.destroyAllWindows()
face_detection.close()
hands.close()
pygame.mixer.quit()
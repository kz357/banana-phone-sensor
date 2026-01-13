import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')

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
    
    # Draw face detection (green box, blue landmarks)
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(
                image_bgr, 
                detection,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Green box
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Blue landmarks
            )
    
    # Draw hand detection (red box, magenta landmarks)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
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
    
    # Draw banana detection (yellow boxes)
    for result in yolo_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                
                # Draw yellow bounding box for banana
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                
                # Add confidence label
                label = f'Banana: {confidence:.2f}'
                cv2.putText(image_bgr, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Display the image
    cv2.imshow('Face, Hand, and Banana Detection', image_bgr)
    
    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
face_detection.close()
hands.close()
import cv2
import mediapipe as mp

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create detection objects
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
    
    # Convert BGR to RGB (MediaPipe uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable
    image_rgb.flags.writeable = False
    
    # Process the image for faces and hands
    face_results = face_detection.process(image_rgb)
    hand_results = hands.process(image_rgb)
    
    # Convert back to BGR for OpenCV display
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw face detection boxes (green box, blue landmarks)
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(
                image_bgr, 
                detection,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Green box
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Blue landmarks
            )
    
    # Draw hand detection boxes (red box for hands)
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
    
    # Display the image
    cv2.imshow('Face and Hand Detection', image_bgr)
    
    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
face_detection.close()
hands.close()
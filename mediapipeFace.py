import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create face detection object
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range (2m), 1 for full-range (5m)
    min_detection_confidence=0.5
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
    
    # Process the image
    results = face_detection.process(image_rgb)
    
    # Convert back to BGR for OpenCV display
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw face detection boxes with custom colors
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(
                image_bgr, 
                detection,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Green box (BGR format)
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Blue landmarks (BGR format)
            )
    
    # Display the image
    cv2.imshow('MediaPipe Face Detection', image_bgr)
    
    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
face_detection.close()
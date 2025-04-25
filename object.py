import torch
import cv2

# Load YOLOv5 model (you can also try 'yolov5s6', 'yolov5m', etc.)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Render results on the frame
    results.render()

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

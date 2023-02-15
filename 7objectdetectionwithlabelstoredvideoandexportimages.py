import cv2
import numpy as np
import os

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Load the class labels
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

# Load the input video
cap = cv2.VideoCapture("video.mp4")

# Create the output folder
if not os.path.exists("output"):
    os.makedirs("output")

frame_number = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Prepare the input blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Extract the class index
            class_idx = int(detections[0, 0, i, 1])

            # Get the bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Add the label
            label = f"{class_names[class_idx]}: {confidence:.2f}"
            y = y1 - 15 if y1 > 15 else y1 + 15
            cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image
    cv2.imwrite(f"output/frame_{frame_number}.jpg", frame)
    frame_number += 1


# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import os

# Load the pre-trained object detection model
model = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Define the classes
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Open the video file
cap = cv2.VideoCapture("video.mp4")

# Create an output file to write the detections
f = open("detections.txt", "w")

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Get the height and width of the frame
        height, width = frame.shape[:2]

        # Convert the frame to a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        model.setInput(blob)
        detections = model.forward()

        # Loop through each detection
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                # Get the class index and the bounding box coordinates
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Write the detection to the output file
                f.write("{},{},{},{},{},{}\n".format(cap.get(cv2.CAP_PROP_POS_MSEC), classes[idx], startX, startY, endX, endY))

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                y = startY - 15 if startY > 15 else startY + 15
                cv2.putText(frame, classes[idx], (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Frame", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    else:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
f.close()
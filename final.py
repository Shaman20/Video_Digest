import cv2
import os
import csv
from datetime import datetime
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Create folders to save images
if not os.path.exists('objects'):
    os.mkdir('objects')

if not os.path.exists('csv'):
    os.mkdir('csv')

# Create csv file to save object positions and timestamps
with open('csv/object_positions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'x', 'y', 'w', 'h', 'timestamp'])

counter = 0
roi_folder = 'objects'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Extract region of interest
    roi = frame[340:720, 500:800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)

    # Save object positions and timestamps to csv file
    with open('csv/object_positions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow([id, x, y, w, h, timestamp])

            # Save the object image with unique id in separate folder
            object_img = roi[y:y + h, x:x + w]
            object_folder = os.path.join(roi_folder, f"{id}")
            if not os.path.exists(object_folder):
                os.mkdir(object_folder)
            img_path = os.path.join(object_folder, f"{id}_{counter}.jpg")
            cv2.imwrite(img_path, object_img)

            # Draw object bounding box and id on the original frame
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

    counter += 1

cap.release()
cv2.destroyAllWindows()

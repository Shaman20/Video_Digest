import os
import csv
import time
import cv2
from tracker import *

object_positions = {}

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Create a directory to save the images
output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# # Create a CSV file to save the object positions and timestamps
# csv_file = open("object_positions.csv", "w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["ID", "X", "Y", "Timestamp"])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340: 720,500: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
        # Store the object's position and appearance time in the object_positions dictionary
        object_positions[id] = [x, y, w, h, cap.get(cv2.CAP_PROP_POS_MSEC)]
        
        # Create a directory with the ID as its name
        id_dir = os.path.join(output_dir, str(id))
        if not os.path.exists(id_dir):
            os.mkdir(id_dir)
        
        # Save the corresponding image in the directory
        img_name = f"{id}_{len(os.listdir(id_dir))+1}.jpg"
        img_path = os.path.join(id_dir, img_name)
        cv2.imwrite(img_path, roi[y:y+h, x:x+w])
        
        # Save the object position and timestamp in the CSV file
        output_file = 'object_positions1.csv'

    # Write the object positions to the output file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'X', 'Y', 'Width', 'Height', 'Appearance Time (ms)'])
            for id, pos in object_positions.items():
                writer.writerow([id, pos[0], pos[1], pos[2], pos[3], pos[4]/1000])
        
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

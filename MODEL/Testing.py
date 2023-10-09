import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best_75.pt')

# Load the image
image = cv2.imread('D:\CSE\Semester - 5\Extra\IIT jammu\YOLO\Code\Test_Dataset\Test_Dataset_2\A0484.png')

# Perform inference
# results = model.predict(image, show=True, save=True)
results = model.predict(image, show=True)

# Check if the image is bleeding
if results[0].boxes.shape[0] > 0:
    print('Bleeding Frame')

    # Print the bounded boxes
    for box in results[0].boxes:
        print(box)

else:
    print('Non-Bleeding Frame')
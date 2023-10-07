import os
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Paths to your images and labels folders
images_folder = r'D:\CSE\Semester - 5\Extra\IIT jammu\YOLO\Code\Data\valid\images'
labels_folder = r'D:\CSE\Semester - 5\Extra\IIT jammu\YOLO\Code\Data\valid\labels'

# Initialize lists to store results
true_labels = []
predicted_labels = []

# Initialize lists to store IoU scores
iou_scores = []

# Initialize the YOLOv8 model
model = YOLO("best_75.pt")

# Initialize lists to store detection results
detection_results = []

def calculate_iou(box1_tensor, box2_list, image_width, image_height):
    # Extract values from the tensor
    x1, y1, x2, y2, confidence, class_id = box1_tensor.tolist()

    # Convert values from the list to strings
    class_id2, x2_gt, y2_gt, w2_gt, h2_gt = map(str, box2_list)

    # Convert coordinates to float
    x2_gt, y2_gt, w2_gt, h2_gt = float(x2_gt), float(y2_gt), float(w2_gt), float(h2_gt)

    # Convert the model's bounding box coordinates to YOLO format
    x1, y1, x2, y2 = x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height

    # Calculate the intersection coordinates
    xA = max(x1, x2_gt)
    yA = max(y1, y2_gt)
    xB = min(x1 + x2, x2_gt + w2_gt)
    yB = min(y1 + y2, y2_gt + h2_gt)

    # Calculate the intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the areas of the boxes
    box1Area = x2 * y2
    box2Area = w2_gt * h2_gt

    # Calculate the IoU
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

# Iterate through image files and corresponding label files
for image_file in tqdm(os.listdir(images_folder)):
    if image_file.endswith('.png'):
        image_path = os.path.join(images_folder, image_file)

        # Determine class (bleeding or non-bleeding) from image file name
        is_bleeding = image_file.startswith('img-')

        # Load the image
        image = cv2.imread(image_path)

        # Perform inference
        results = model.predict(image)

        # Check if bleeding is detected
        detected_bleeding = results[0].boxes.shape[0] > 0

        # Append true and predicted labels
        true_labels.append(is_bleeding)
        predicted_labels.append(detected_bleeding)

        # Extract ground truth bounding boxes from label file
        label_file_name = os.path.splitext(image_file)[0] + '.txt'
        label_file_path = os.path.join(labels_folder, label_file_name)

        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as label_file:
                lines = label_file.readlines()
                ground_truth_boxes = [list(map(float, line.strip().split())) for line in lines]

            # Calculate IoU for each detected box
            if results[0].boxes.shape[0] > 0:
                for box in results[0].boxes.boxes:
                    iou_values = []
                    for gt_box in ground_truth_boxes:
                        iou = calculate_iou(box, gt_box, 224, 224)
                        iou_values.append(iou)

                    max_iou = max(iou_values) if iou_values else 0
                    iou_scores.append(max_iou)
            
            # Append detection results
            if results[0].boxes.shape[0] > 0:
                box = results[0].boxes[0]
                detection_results.append({
                    'image_path': image_path,
                    'is_bleeding': is_bleeding,
                    'detection_confidence': box.conf[0].item()
                })

# Calculate classification metrics
accuracy = accuracy_score(true_labels, predicted_labels)
recall = classification_report(true_labels, predicted_labels, target_names=['Non-Bleeding', 'Bleeding'], output_dict=True)
f1 = f1_score(true_labels, predicted_labels)

# Create a table of achieved evaluation metrics
classification_metrics = {
    'Accuracy': [accuracy],
    'Recall (Non-Bleeding)': [recall['Non-Bleeding']['recall']],
    'Recall (Bleeding)': [recall['Bleeding']['recall']],
    'F1-Score': [f1]
}

classification_df = pd.DataFrame(classification_metrics)
print("Classification Metrics:")
print(classification_df)

print()
print("Detection Metrics:")
print("IoU Metrics:")
# Calculate and print IoU statistics
iou_scores = np.array(iou_scores)
iou_mean = np.mean(iou_scores)
iou_median = np.median(iou_scores)
iou_75th_percentile = np.percentile(iou_scores, 75)
print("IoU Mean:", iou_mean)
print("IoU Median:", iou_median)
print("IoU 75th Percentile:", iou_75th_percentile)

# Calculate Detection Average Precision (AP) and Mean Average Precision (mAP)
detection_results_df = pd.DataFrame(detection_results)
detection_results_df['detection_confidence'] = detection_results_df['detection_confidence'].apply(lambda x: x.item() if isinstance(x, np.float32) else x)
detection_results_df['is_detected'] = detection_results_df['detection_confidence'] > 0.5  # You can adjust the confidence threshold as needed
detection_results_df['is_detected'] = detection_results_df['is_detected'].astype(int)
detection_ap = detection_results_df.groupby('is_bleeding')['is_detected'].apply(lambda x: np.mean(x))
mAP = detection_ap.mean()
print("Detection Average Precision (AP):")
print(detection_ap)
print("Mean Average Precision (mAP):", mAP)

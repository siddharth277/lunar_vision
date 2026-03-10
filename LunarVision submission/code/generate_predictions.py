import torch
import os
import cv2

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Input and output folder paths
test_folder = 'test_images'
output_folder = 'predicted_labels'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all test images
for filename in os.listdir(test_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(test_folder, filename)
    image = cv2.imread(image_path)

    # Run YOLOv5 inference
    results = model(image, size=640)
    predictions = results.xywh[0]  # x_center, y_center, width, height

    # Save to predicted_labels/filename_label.txt
    base_name = os.path.splitext(filename)[0]
    label_filename = f"{base_name}_label.txt"
    label_path = os.path.join(output_folder, label_filename)

    with open(label_path, 'w') as f:
        for *xywh, conf, cls in predictions.tolist():
            class_id = int(cls)
            x_center, y_center, width, height = [round(val, 6) for val in xywh]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ All predictions saved in predicted_labels/")
import gradio as gr
import torch
from PIL import Image
import cv2
import numpy as np

# Load your trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Detection Function
def detect_moon_features(img):
    # Save original image
    original_img = np.array(img)

    # Convert to grayscale and enhance contrast
    img_gray = img.convert("L")
    img_eq = cv2.equalizeHist(np.array(img_gray))
    img_for_yolo = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

    # Run detection
    results = model(img_for_yolo)
    detections_df = results.pandas().xyxy[0]

    if detections_df.empty:
        return img, img, "No craters or boulders detected."

    # Copies for drawing boxes
    processed_img = img_for_yolo.copy()
    original_with_boxes = original_img.copy()

    for _, row in detections_df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        color = (255, 0, 0) if label == 'crater' else (0, 0, 255)  # blue = crater, red = boulder

        # Draw on both images
        cv2.rectangle(original_with_boxes, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 2)

    # Count features
    crater_count = sum(d == 'crater' for d in detections_df['name'])
    boulder_count = sum(d == 'boulder' for d in detections_df['name'])

    return Image.fromarray(original_with_boxes), Image.fromarray(processed_img), f"🕳 {crater_count} Craters, 🪨 {boulder_count} Boulders"

# Gradio UI
gr.Interface(
    fn=detect_moon_features,
    inputs=gr.Image(type="pil", label="Upload Lunar Image"),
    outputs=[
        gr.Image(type="pil", label="🖼 Original Image (with Boxes)"),
        gr.Image(type="pil", label="⚙ Processed Image (with Boxes)"),
        gr.Textbox(label="Detection Count")
    ],
    title="Lunar Crater & Boulder Detector",
    description=(
        "Detects craters and boulders on the moon's surface using a trained YOLOv5 model.<br><br>"
        "<b>Legend:</b><br>"
        "🟥 <span style='color:red;'>Red box</span> = Boulder<br>"
        "🟦 <span style='color:blue;'>Blue box</span> = Crater"
    )
).launch()
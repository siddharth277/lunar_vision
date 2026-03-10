#  Lunar Vision Challenge 2025 – Crater & Boulder Detection

This project is our submission for the SOI Lunar Vision Challenge. We built an AI-based system to detect craters and boulders on lunar surface images using YOLOv5. The project includes model training, test label prediction, and an interactive Gradio-based UI to visualize the detections.

---

##  Objective

Detect and localize craters and boulders on lunar terrain using computer vision techniques, and output predictions in YOLO format for a test set of images.

---

##  Summary of Our Approach

1. Data Preparation
   - Organized the dataset into `train/` and `valid/` with separate `images/` and `labels/` folders.
   - Created a custom `moon.yaml` to configure training for 2 classes: `crater` and `boulder`.

2. Model Training
   - Used YOLOv5s for speed and efficiency.
   - Trained locally in VS Code using optimized settings 
   - Applied preprocessing (histogram equalization) to boost crater visibility.

3. Test Label Prediction
   - Built a script (`generate_predictions.py`) to generate YOLO-format `.txt` label files for the test set.
   - Saved each file as `filename_label.txt` as required.

4. Gradio UI
   - Built a clean interface to upload images and view detection results.
   - Displays both original and processed (contrast-enhanced) image outputs.
   - Counts and displays number of craters and boulders detected.

---

##  Folder Structure

LunarVision_Submission/
├── predicted_labels/ # YOLO-format labels for test images
├── code/
│ ├── ui.py # Gradio UI with dual image output
│ ├── generate_predictions.py # Script to generate test labels
│ ├── moon.yaml # Data configuration file for YOLOv5
├── best.pt # Trained YOLOv5 model
├── requirements.txt # List of Python dependencies
├── README.md # This file
├── Lunar_Report.pdf # 5-page project summary report


---

##  How to Run the Project

1. Setup Environment

Install Python (3.8+)  
Then install required libraries:

pip install -r requirements.txt


2. Launch the UI
code:
	python ui.py

A browser window will open at http://127.0.0.1:7860.
Upload a lunar image and view the detection results on both:

Original image
Contrast-enhanced processed image

Also shows the number of craters and boulders detected.


3. Generate Predicted Labels for Test Set

Place all test images in a folder named test_images/ inside the main directory.

Run the following command:
code :
	python generate_predictions.py
This will create a folder predicted_labels:
moon_01_label.txt
moon_02_label.txt
...
Each file follows the YOLO format and naming convention required for submission.

YOLO Training Configuration

Base model: YOLOv5s
Number of classes: 2 (crater, boulder)
Image size: 640
Batch size: 16
Epochs: 5

Training command used:

python train.py --img 640 --batch 16 --epochs 5 --data moon.yaml --weights yolov5s.pt --name lunar_detector


## Notes

crater is class 0
boulder is class 1



## Final Words

Thanks for the challenge!! 

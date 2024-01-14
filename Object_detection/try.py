import torch
from matplotlib import pyplot as plt
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 'yolov5s' is the smallest model. You can also use 'yolov5m', 'yolov5l', or 'yolov5x'.

# Load an image
img_path = 'Object_detection\\image.jpg'  # Replace with your image path
img = cv2.imread(img_path)  # Using OpenCV to load the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Perform inference
results = model(img)

# Results
results.print()  # Print results to console
results.show()  # Show the image with bounding boxes



# To use the results in your application, you can access the raw predictions
# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)


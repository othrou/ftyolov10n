from ultralytics import YOLO
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Load the exported TFLite model
model = YOLO("best_float32.tflite")

# Fetch the image from the URL
image_url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Convert image to numpy array
image_np = np.array(image)

# Run inference
results = model(image_np)

# Display the results
for item in results:
    item.show()

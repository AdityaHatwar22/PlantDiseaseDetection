import tensorflow as tf
import cv2
import numpy as np
import os

# Check model exists
if not os.path.exists("model.h5"):
    print("❌ Model not found! Run train.py first")
    exit()

# Load model
model = tf.keras.models.load_model("model.h5")

# Check image exists
if not os.path.exists("test.jpg"):
    print("❌ test.jpg not found")
    exit()

# Read image
img = cv2.imread("test.jpg")

if img is None:
    print("❌ Image not loaded. Check path or file.")
    exit()

# Preprocess
img = cv2.resize(img,(224,224))
img = img / 255.0
img = img.reshape(1,224,224,3)

# Predict
pred = model.predict(img)

classes = ["Diseased", "Healthy"]
result = classes[np.argmax(pred)]

print("🌿 Prediction:", result)
print("📊 Confidence:", round(100 * np.max(pred), 2), "%")
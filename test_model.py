# Load and test the model on 16 sample images

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
from imutils import build_montages
import cv2
import numpy as np
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True)
args = vars(ap.parse_args())

model = load_model("covid19_model.h5")

imagePaths  = list(paths.list_images(args["images"]))
random.shuffle(imagePaths) # 16 random samples
imagePaths = imagePaths[:16]


results = []
for i in imagePaths:
    orig = cv2.imread(i)
    
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float") / 255.0

    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    pred = pred.argmax(axis=1)[0]

    label = "negative" if pred == 0 else "positive"
    color = (0, 0, 255) if pred == 0 else (0, 255, 0)

    orig = cv2.resize(orig, (128, 128))
    cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    results.append(orig)


montage = build_montages(results, (128, 128), (4, 4))[0]
cv2.imshow("Results", montage)
cv2.waitKey(0)

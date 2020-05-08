from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="path to input dataset")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
#                 help="path to output loss/accuracy plot")
# ap.add_argument("-m", "--model", type=str, default="covid19.model",
#                 help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 25
BS = 8

imagepaths = list(paths.list_images("images"))
data = []
labels = []

i = 0
for imgpath in imagepaths:
    label = imgpath.split(os.path.sep)[-2]

    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    data.append(img)
    labels.append(label)

data = np.array(data)
data /= 255.0
labels = np.array(labels)

lblbin  = LabelBinarizer()
labels = lblbin.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2, stratify=labels, random_state=3)

trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for i in baseModel.layers:
    i.trainable = False;

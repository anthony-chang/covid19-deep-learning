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

INIT_LR = 1e-3
EPOCHS = 25
BS = 8

print("loading images")
imagepaths = list(paths.list_images("images"))
data = []
labels = []

for imgpath in imagepaths:
    label = imgpath.split(os.path.sep)[-2]

    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    data.append(img)
    labels.append(label)

data = np.array(data) / 255.0
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


print("compiling model")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("training head")
H = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=BS),
                        steps_per_epoch=len(trainX) // BS,
                        validation_data=(testX, testY),
                        validation_steps=len(testX)// BS,
                        epochs=EPOCHS)

print("evaluating model")
predictInd = model.predict(testX, batch_size=BS)
predictInd = np.argmax(predictInd, axis=1)

print(classification_report(testY.argmax(axis=1), predictInd, target_names=lblbin.classes_))

conf_mat = confusion_matrix(testY.argmax(axis=1), predictInd)
total = sum(sum(conf_mat))
acc = (conf_mat[0, 0] + conf_mat[1, 1]) / total
sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
specificity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])

print(conf_mat)
print("acc: {:f}".format(acc))
print("sensitivity: {:f}".format(sensitivity))
print("specificity: {:f}".format(specificity))


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss/Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

print("saving model")
model.save("covid19.model", save_format="h5")

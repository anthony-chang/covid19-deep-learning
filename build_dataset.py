import pandas as pd
import argparse
import shutil
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--covid", required=True,
#                 help="path covid19 dataset")
# ap.add_argument("-o", "--output", required=True,
#                 help="path to where images is stored")
# args = vars(ap.parse_args())

df = pd.read_csv("covid-chestxray-dataset\metadata.csv")

for (i, row) in df.iterrows():
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    imgpath = os.path.sep.join(["covid-chestxray-dataset\images", row["filename"]])
    
    if not os.path.exists(imgpath):
        print("error file does not exist")
        continue

    filename = row["filename"].split(os.path.sep)[-1]
    outputpath = "images"

    shutil.copy2(imgpath, outputpath)

    

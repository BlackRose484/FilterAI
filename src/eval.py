import pandas as pd
import numpy as np
import cv2 as cv
import torch
from src.Model import EfficientNetB0
from src.filter import *

# Create Model
model = EfficientNetB0(68)
model.load_state_dict(torch.load('../data/models/best_model_3.pth'))

# Get image, filter
image = cv.imread("../data/kaggle/working/ibug_300W_large_face_landmark_dataset/helen/trainset/232194_1.jpg")
# Change filter_image and filter_csv to your own
filter_image = cv.imread("../data/filter/images/squid.png")
filter_csv = "../data/filter/csv/squid.csv"

# Create Filter App
filterApp = Filter(model=model)

# Filter An Image
image_filter = filterApp.filter_image(img=image, filter=filter_image, filter_csv=filter_csv)
image_filter = cv.resize(image_filter, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow("Image", image_filter)

# Filter With camera
filterApp.filter_camera(filter=filter_image, filter_csv=filter_csv)

cv.waitKey(0)
cv.destroyAllWindows()
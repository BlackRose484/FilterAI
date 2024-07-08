import dlib
from PIL import Image, ImageDraw
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform_image = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dectector = dlib.get_frontal_face_detector()

def get_face_image(image):
    faces = dectector(np.array(image))
    print(len(faces))
    for face in faces:
        top, left, right, bottom, width, height = face.top(), face.left(), face.right(), face.bottom(), face.width(), face.height()
        box = (left, top, right , bottom)
        image_crop = image.crop(box)
        image = transform_image(image_crop)

        return image, image_crop, np.array([width, height], dtype=np.float32)

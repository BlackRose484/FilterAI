import cv2 as cv
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from src.Model.EfficientNetB0 import *
import dlib
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


transform_pred = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Filter():
    def __init__(self, model, image=None, img_path=None, filter_path=None):
        self.img_path = img_path
        self.filter_path = filter_path
        self.model = model.to(device)
        self.face_detect = dlib.get_frontal_face_detector()
        self.filter_landmarks = None
        self.triangle_list = None

    def detect_landmark(self, image):
        self.model.eval()

        # Covert BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h, w, c = image.shape
        # Process image to suitable for model
        process_image = transform_pred(image=image)
        # plt.imshow(process_image['image'].cpu().clone().permute(1, 2, 0))
        process_image = process_image['image'].unsqueeze(0)
        process_image = process_image.to(device)

        # Get output from model
        output = self.model(process_image)
        output = output.view(68, 2)

        landmarks = (output + 0.5)
        landmarks = landmarks.detach().cpu().numpy()
        landmarks[:, 0] = landmarks[:, 0] * w
        landmarks[:, 1] = landmarks[:, 1] * h

        # landmarks = np.append(landmarks, [[x + 50, y]], axis=0)
        # landmarks = np.append(landmarks, [[x + w - 50, y]], axis=0)
        return landmarks
    def get_filter_landmarks_and_delaunay_triangle(self, filter):
        landmarks = []
        indexes_triangles = []
        img_gray = cv.cvtColor(filter, cv.COLOR_BGR2GRAY)
        faces = self.face_detect(img_gray)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            landmarks = self.detect_landmark(filter[y: y + h, x : x + w])
            landmarks += np.array([x, y])
            points = np.array(landmarks, dtype=np.int32)

            # Delaunay Area
            convexhull = cv.convexHull(points)
            bounding_box = cv.boundingRect(convexhull)
            subdiv2 = cv.Subdiv2D(bounding_box)
            subdiv2.insert(landmarks)
            triangles = subdiv2.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                id_pt1 = np.where((points == pt1).all(axis=1))
                id_pt1 = extract_index_nparray(id_pt1)

                id_pt2 = np.where((points == pt2).all(axis=1))
                id_pt2 = extract_index_nparray(id_pt2)

                id_pt3 = np.where((points == pt3).all(axis=1))
                id_pt3 = extract_index_nparray(id_pt3)

                if id_pt1 is not None and id_pt2 is not None and id_pt3 is not None:
                    triangle = [id_pt1, id_pt2, id_pt3]
                    indexes_triangles.append(triangle)
        return landmarks, indexes_triangles

    def apply_filter(self, image, filter, image_landmarks, filter_landmarks, triangle_list):
        img2 = image
        img = filter
        img2_new_face = np.zeros_like(img2)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        landmarks_points = np.array(filter_landmarks, dtype=np.int32)
        landmarks_points2 = np.array(image_landmarks, dtype=np.int32)
        convex_hull = cv.convexHull(landmarks_points)
        convex_hull2 = cv.convexHull(landmarks_points2)
        for triangle_index in triangle_list:
            # Face 1
            # Buoc 1: Xac dinh tung tam giac
            tri_pt1 = landmarks_points[triangle_index[0]]
            tri_pt2 = landmarks_points[triangle_index[1]]
            tri_pt3 = landmarks_points[triangle_index[2]]
            triangle = np.array([tri_pt1, tri_pt2, tri_pt3], dtype=np.int32)

            # Buoc 2: Xac dinh vi tri va phan bounding cua moi tam giac
            bounding_rect1 = cv.boundingRect(triangle)
            (x, y, w, h) = bounding_rect1
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            crop_image1 = img[y:y + h, x:x + w]
            crop_image1_mask = np.zeros((h, w), dtype=np.uint8)
            points = np.array([[tri_pt1[0] - x, tri_pt1[1] - y],
                               [tri_pt2[0] - x, tri_pt2[1] - y],
                               [tri_pt3[0] - x, tri_pt3[1] - y]], np.int32)

            # Buoc 3: Tao ra 1 mask giup xac dinh chinh xac vi tri tam giac duoc chuyen doi de khong anh huong den cac pixel khac
            cv.fillConvexPoly(crop_image1_mask, points, 255)
            try:
                crop_image1 = cv.bitwise_and(crop_image1, crop_image1, mask=crop_image1_mask)
            except:
                print("Loi crop_image1")
            # cv.line(img, tri_pt1, tri_pt2, (0, 0, 255), 2)
            # cv.line(img, tri_pt3, tri_pt2, (0, 0, 255), 2)
            # cv.line(img, tri_pt1, tri_pt3, (0, 0, 255), 2)

            # Face 2
            tri2_pt1 = landmarks_points2[triangle_index[0]]
            tri2_pt2 = landmarks_points2[triangle_index[1]]
            tri2_pt3 = landmarks_points2[triangle_index[2]]

            triangle2 = np.array([tri2_pt1, tri2_pt2, tri2_pt3], dtype=np.int32)

            bounding_rect2 = cv.boundingRect(triangle2)
            (x, y, w, h) = bounding_rect2
            # cv.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
            crop_image2 = img2[y:y + h, x:x + w]
            crop_image2_mask = np.zeros((h, w), dtype=np.uint8)
            points2 = np.array([[tri2_pt1[0] - x, tri2_pt1[1] - y],
                                [tri2_pt2[0] - x, tri2_pt2[1] - y],
                                [tri2_pt3[0] - x, tri2_pt3[1] - y]], np.int32)
            cv.fillConvexPoly(crop_image2_mask, points2, 255)
            try:
                crop_image2 = cv.bitwise_and(crop_image2, crop_image2, mask=crop_image2_mask)
            except:
                print("Loi crop_image2")

            # cv.line(img2, tri2_pt1, tri2_pt2, (0, 0, 255), 2)
            # cv.line(img2, tri2_pt3, tri2_pt2, (0, 0, 255), 2)
            # cv.line(img2, tri2_pt1, tri2_pt3, (0, 0, 255), 2)

            cropped_tr2_mask = np.zeros((h, w), np.uint8)
            cv.fillConvexPoly(cropped_tr2_mask, points2, 255)

            points = np.float32(points)
            points2 = np.float32(points2)

            # Buoc 4: Thuc hien chuyen doi giua 2 tam giac
            # getAffine nhan vao chinh xac 3 diem de tao ma tran chuyen doi

            M = cv.getAffineTransform(points, points2)
            crop_trans = cv.warpAffine(crop_image1, M, (w, h), flags=cv.INTER_NEAREST)
            crop_trans = cv.bitwise_and(crop_trans, crop_trans, mask=cropped_tr2_mask)

            img2_new_face_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_area_gray = cv.cvtColor(img2_new_face_area, cv.COLOR_BGR2GRAY)
            _, mask_designed = cv.threshold(img2_new_face_area_gray, 1, 255, cv.THRESH_BINARY_INV)
            try:
                crop_trans = cv.bitwise_and(crop_trans, crop_trans, mask=mask_designed)
            except:
                print("Loi crop_trans")

            img2_new_face_area = cv.add(img2_new_face_area, crop_trans)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_area

        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv.fillConvexPoly(img2_face_mask, convex_hull2, 255)
        img2_face_mask = cv.bitwise_not(img2_head_mask)

        try:
            img2_noface = cv.bitwise_and(img2, img2, mask=img2_face_mask)
        except:
            print("Loi im2_noface")
        result = cv.add(img2_noface, img2_new_face)

        # Adjust color
        (x, y, w, h) = cv.boundingRect(convex_hull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv.seamlessClone(result, img2, img2_head_mask, center_face2, cv.NORMAL_CLONE)
        # seamlessclone = cv.resize(seamlessclone, (500, 500), interpolation=cv.INTER_AREA)
        # cv.imshow("Image", img)
        # cv.imshow("Image_Face 1", face_image_1)
        # cv.imshow("Image2", img2)
        # cv.imshow("Image_Face 2", face_image_2)
        # cv.imshow("Crop Image 1", crop_image1)
        # cv.imshow("Crop Image 2", crop_image2)
        # cv.imshow("Crop Transform", crop_trans)
        # cv.imshow("New face", img2_new_face)
        # cv.imshow("Result", seamlessclone)
        # cv.waitKey(0)
        return seamlessclone

    def camera_filter(self, filter):
        cap = cv.VideoCapture(0)
        while True:
            _, frame = cap.read()
            frame = self.filter_image(img=frame, filter=filter)

            cv.imshow("Camera Filter", frame)

            if cv.waitKey(10) & 0xFF == ord('d'):
                break

    def filter_image(self, img, filter):
        faces = self.face_detect(img)
        if self.filter_landmarks is None and self.triangle_list is None:
            self.filter_landmarks, self.triangle_list = self.get_filter_landmarks_and_delaunay_triangle(filter)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            image_landmarks = self.detect_landmark(img[y: y + h, x: x + w])
            image_landmarks += np.array([x, y])

            img = self.apply_filter(img, filter, image_landmarks, self.filter_landmarks, self.triangle_list)

        return img


model = EfficientNetB0(68)
model.load_state_dict(torch.load('best_model_3.pth'))
filter_image = cv.imread("joker.jfif")
image = cv.imread("../data/kaggle/working/ibug_300W_large_face_landmark_dataset/helen/testset/296961468_1_mirror.jpg")
filter = Filter(model)
result = filter.filter_image(image, filter_image)
result = cv.resize(result, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow("Result", result)
cv.waitKey(0)
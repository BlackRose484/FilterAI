import cv2
import numpy as np
import cv2 as cv
import dlib

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


img = cv2.imread("joker.jfif")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")

indexes_triangles = []

# Face 1
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        landmarks_points.append((x, y))
        # cv.circle(img, (x, y), 2, (0, 255, 0), 2)
    points = np.array(landmarks_points, dtype=np.int32)

    # Delaunay Area
    convexhull = cv.convexHull(points)
    bounding_box = cv.boundingRect(convexhull)
    subdiv2 = cv.Subdiv2D(bounding_box)
    subdiv2.insert(landmarks_points)
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

# Init camera
cap = cv.VideoCapture(0)
while True:
    _, img2 = cap.read()
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img2_new = np.zeros_like(img2)
    faces = detector(img2_gray)

    # Detect face and landmarks
    for face in faces:
        landmarks2 = predictor(img2_gray, face)
        landmarks_points2 = []
        for i in range(68):
            x = landmarks2.part(i).x
            y = landmarks2.part(i).y

            landmarks_points2.append((x, y))
            # cv.circle(img2, (x, y), 2, (0, 255, 0), 1)

        points2 = np.array(landmarks_points2, dtype=np.int32)
        convexhull2 = cv.convexHull(points2)

        # Create Triangle
        for t in indexes_triangles:
            # delaunay triangle 1
            tri_pt1 = landmarks_points[t[0]]
            tri_pt2 = landmarks_points[t[1]]
            tri_pt3 = landmarks_points[t[2]]
            triangle1 = np.array([tri_pt1, tri_pt2, tri_pt3], dtype=np.int32)
            bounding_rect1 = cv.boundingRect(triangle1)
            (x, y, w, h) = bounding_rect1
            crop_area1 = img[y: y + h, x: x + w]
            crop_area1_mask = np.zeros((h, w), dtype=np.uint8)
            points_in_crop = np.array([[tri_pt1[0] - x, tri_pt1[1] - y],
                                       [tri_pt2[0] - x, tri_pt2[1] - y],
                                       [tri_pt3[0] - x, tri_pt3[1] - y]], np.int32)
            cv.fillConvexPoly(crop_area1_mask, points_in_crop, 255)
            try:
                crop_area1 = cv.bitwise_and(crop_area1, crop_area1, mask=crop_area1_mask)
            except:
                pass

            # delaunay triangle 2
            tri2_pt1 = landmarks_points2[t[0]]
            tri2_pt2 = landmarks_points2[t[1]]
            tri2_pt3 = landmarks_points2[t[2]]

            triangle2 = np.array([tri2_pt1, tri2_pt2, tri2_pt3], dtype=np.int32)
            bounding_rect2 = cv.boundingRect(triangle2)
            (x, y, w, h) = bounding_rect2
            crop_area2 = img2[y: y + h, x: x + w]
            crop_area2_mask = np.zeros((h, w), dtype=np.uint8)

            points2_in_crop = np.array([[tri2_pt1[0] - x, tri2_pt1[1] - y],
                                        [tri2_pt2[0] - x, tri2_pt2[1] - y],
                                        [tri2_pt3[0] - x, tri2_pt3[1] - y]], np.int32)
            cv.fillConvexPoly(crop_area2_mask, points2_in_crop, 255)
            try:
                crop_area2 = cv.bitwise_and(crop_area2, crop_area2, mask=crop_area2_mask)
            except:
                pass

            # cv.line(img2, tri2_pt1, tri2_pt2, (0, 255, 0), 1)
            # cv.line(img2, tri2_pt2, tri2_pt3, (0, 255, 0), 1)
            # cv.line(img2, tri2_pt3, tri2_pt1, (0, 255, 0), 1)

            # Convert from delaunay 1 to delaunay 2
            points_in_crop = np.float32(points_in_crop)
            points2_in_crop = np.float32(points2_in_crop)

            try:
                M = cv.getAffineTransform(points_in_crop, points2_in_crop)
                crop_trans = cv.warpAffine(crop_area1, M, (w, h), flags=cv.INTER_NEAREST)
                crop_trans = cv.bitwise_and(crop_trans, crop_trans, mask=crop_area2_mask)

                img2_new_area = img2_new[y: y + h, x: x + w]
                img2_new_area_gray = cv.cvtColor(img2_new_area, cv.COLOR_BGR2GRAY)
                _, mask_designed = cv.threshold(img2_new_area_gray, 1, 255, cv.THRESH_BINARY_INV)
                crop_trans = cv.bitwise_and(crop_trans, crop_trans, mask=mask_designed)

                img2_new_area = cv.add(crop_trans, img2_new_area)
                img2_new[y: y + h, x: x + w] = img2_new_area
            except:
                pass

        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv.bitwise_not(img2_head_mask)

        img2_noface = cv.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv.add(img2_noface, img2_new)

        (x, y, w, h) = cv.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        try:
            img2 = cv.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
        except:
            pass


    cv.imshow("Camera", img2)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()

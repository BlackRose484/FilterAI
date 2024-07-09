import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
from tqdm import tqdm

transform_train = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Resize(height=256, width=256, always_apply=True),
    A.RandomCrop(height=224, width=224, always_apply=True),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
#     A.Cutout(num_holes=8, max_h_size=18, max_w_size=18, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


transform_test = A.Compose([
    A.Resize(height=256, width=256, always_apply=True),
    A.CenterCrop(height=224, width=224, always_apply=True),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class DataAugment():
    def __init__(self):
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.images = None
        self.landmarks = None

    def norm_transform_dataset(self, images, landmarks, transform):
        final_images = []
        final_landmarks = []

        for i in tqdm(range(len(images))):
            img = images[i]
            ldm = landmarks[i]

            img = np.array(img)

            # transform
            transformed = transform(image=img, keypoints=ldm)
            transformed_img = transformed['image']
            transformed_lmd = transformed['keypoints']

            # normalize
            color_channels, height, width = transformed_img.shape
            transformed_lmd = transformed_lmd / np.array([width, height]) - 0.5
            transformed_lmd = torch.tensor(transformed_lmd, dtype=torch.float32)

            final_images.append(transformed_img)
            final_landmarks.append(transformed_lmd)

        self.images = final_images
        self.landmarks = final_landmarks

        return final_images, final_landmarks

    def finish_data_tensor(self, images, landmarks):
        final_images_train_converted = torch.stack([tensor.permute(1, 2, 0) for tensor in images])
        final_landmarks_train = np.array(landmarks)

        return final_images_train_converted, final_landmarks_train


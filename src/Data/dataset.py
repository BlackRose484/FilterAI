import xml.etree.ElementTree as ET
import numpy as np
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

class IDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images, self.landmarks = self.generateData(self.data_dir)

    def sample_image(self, image):
        """
        Create a dictionary containing image details.

        Args:
            image: An XML element containing image data.

        Returns:
            A dictionary with image attributes including:
            - filename (str)
            - width (int)
            - height (int)
            - box_top (int)
            - box_left (int)
            - box_width (int)
            - box_height (int)
            - landmarks (np.ndarray of tuples): Each tuple contains (x, y) coordinates of a landmark.
        """

        image_result = {}
        image_result['filename'] = image.attrib['file']
        image_result['width'] = int(image.attrib['width'])
        image_result['height'] = int(image.attrib['height'])

        box = image.find('box')
        image_result['box_top'] = int(box.attrib['top'])
        image_result['box_left'] = int(box.attrib['left'])
        image_result['box_width'] = int(box.attrib['width'])
        image_result['box_height'] = int(box.attrib['height'])

        # set up landmarks
        landmarks = np.array([[float(part.attrib["x"]), float(part.attrib["y"])] for part in box])
        image_result['landmarks'] = landmarks

        return image_result

    def create_samples_xml(self, xml_file_path):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        images = root.find('images')
        samples: list[dict] = [self.sample_image(image) for image in images]
        return samples

    def get_data(self, samples, root_dir):
        images = []
        landmarks_list = []
        for sample in samples:
            image_path = os.path.join(root_dir, sample['filename'])
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                width = sample['width']
                height = sample['height']
                box_left = sample["box_left"]
                box_top = sample["box_top"]
                box_width = sample["box_width"]
                box_height = sample["box_height"]
                landmarks = sample['landmarks']
                crop_image = image.crop((box_left, box_top, box_left + box_width, box_top + box_height))
                landmarks -= np.array([box_left, box_top])
                images.append(crop_image)
                landmarks_list.append(landmarks)
        return images, landmarks_list

    def generateData(self, path):
        samples = self.create_samples_xml(path)
        images, landmarks = self.get_data(samples, os.path.dirname(path))
        return images, landmarks

class LandmarkData(Dataset):
    def __init__(self, images: torch.Tensor, landmarks: torch.Tensor):
        self.images = images
        self.landmarks = landmarks

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.images[idx], self.landmarks[idx]



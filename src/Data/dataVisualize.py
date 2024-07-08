from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def visualize_image(sample, root_dir):
    # Open the image
    image_path = os.path.join(root_dir, sample['filename'])
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        # Get image dimensions and box coordinates
        width = sample['width']
        height = sample['height']
        box_left = sample["box_left"]
        box_top = sample["box_top"]
        box_width = sample["box_width"]
        box_height = sample["box_height"]
        landmarks = sample['landmarks']

        # Draw bounding box
        draw = ImageDraw.Draw(image)
        box = [box_left, box_top, box_left + box_width, box_top + box_height]
        draw.rectangle(box, outline="red", width=3)

        # Draw landmarks
        for (x, y) in landmarks:
            draw.ellipse((x-2, y-2, x+2, y+2), fill='blue', outline='blue')

        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()

def visualize_data(image,landmark):
    image_clone = image.copy()
    draw = ImageDraw.Draw(image_clone)
    for (x, y) in landmark:
        draw.ellipse((x-2, y-2, x+2, y+2), fill='blue', outline='blue')

    # Display the image
    plt.imshow(image_clone)
    plt.axis('off')  # Hide axes
    plt.show()


def visualize_final(transformed_cropped_image, normalized_transformed_landmarks):
    # Define mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Reverse color transform
    def denormalize(image_tensor, mean=mean, std=std):
        # Clone and permute the tensor
        tmp = image_tensor.clone().permute(1, 2, 0)

        # Denormalize
        for t, m, s in zip(tmp, mean, std):
            t.mul_(s).add_(m)

        # Clamp the values
        return torch.clamp(tmp, 0, 1)

    cropped_image = denormalize(transformed_cropped_image)

    # Get size of cropped image
    cropped_image = (cropped_image.numpy() * 255).astype(np.uint8)
    height, width, color_channels = cropped_image.shape

    # Denormalize landmarks to pixel coordinates
    landmarks = (normalized_transformed_landmarks + 0.5) * np.array([width, height])

    visualize_data(Image.fromarray(cropped_image.astype(np.uint8)).convert('RGB'), landmarks)
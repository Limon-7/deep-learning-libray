import os
import albumentations as A
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from utils import plot_examples

path = Path(__file__).parent / "images/elon.jpeg"
mask_path = Path(__file__).parent / "images/mask.jpeg"
mask_path2 = Path(__file__).parent / "images/second_mask.jpeg"

print(mask_path2)

image = Image.open(path)
mask = Image.open(mask_path)
mask1 = Image.open(mask_path2)

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ],
            p=1.0,
        ),
    ], is_check_shapes=False,
)

image_list = [image]
image = np.array(image)
mask = np.array(mask)
mask1 = np.array(mask1)

for i in range(4):
    augmentations = transform(image=image, masks= [mask,mask1])
    augmented_image = augmentations["image"]
    augmented_mask = augmentations["masks"]
    image_list.append(augmented_image)
    image_list.append(augmented_mask[0])
    image_list.append(augmented_mask[1])

plot_examples(image_list)

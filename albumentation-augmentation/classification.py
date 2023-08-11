import os
import albumentations as A
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from utils import plot_examples

path = Path(__file__).parent / "images/cat.jpg"
print(path)

# image = cv2.imread(path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = Image.open(path)

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
    ]
)

image_list = [image]
image = np.array(image)
for i in range(10):
    augmentations = transform(image=image)
    augmented_image = augmentations["image"]
    image_list.append(augmented_image)

plot_examples(image_list)

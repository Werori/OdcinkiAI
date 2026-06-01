import os
import random

import cv2
import numpy as np

NUMBER_OF_IMAGES = 1000
WHITE = 255
MAX_NUMBER_OF_LINES = 20


def generate_data_for_labels(
    class_id: int, x1: int, x2: int, y1: int, y2: int, resolution: tuple[int, int]
) -> tuple[int, float, float, float, float]:
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    image_width, image_height = resolution
    x_center = (x_min + x_max) / 2
    x_center_norm = x_center / image_width
    box_width_norm = (x_max - x_min) / image_width
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    y_center = (y_min + y_max) / 2
    y_center_norm = y_center / image_height
    box_height_norm = (y_max - y_min) / image_height
    return (class_id, x_center_norm, y_center_norm, box_width_norm, box_height_norm)


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.01) -> np.ndarray:
    noisy_image = image.copy()
    height, width, _ = image.shape
    num_pixels = (
        height * width
    )  # Rzeczywista liczba pikseli (bez mnożenia przez 3 kanały)

    num_salt = int(np.ceil(amount * num_pixels * 0.5))
    num_pepper = int(np.ceil(amount * num_pixels * 0.5))

    # Dodawanie soli (białe piksele)
    for _ in range(num_salt):
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        noisy_image[y, x] = [255, 255, 255]

    # Dodawanie pieprzu (czarne piksele)
    for _ in range(num_pepper):
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        noisy_image[y, x] = [0, 0, 0]

    return noisy_image


os.makedirs("daneTreningowe/images", exist_ok=True)
os.makedirs("daneTreningowe/labels", exist_ok=True)

resolutions = [(640, 480), (320, 240), (160, 120)]

for i in range(0, NUMBER_OF_IMAGES):
    random_resolution = resolutions[random.randint(0, len(resolutions) - 1)]

    img = np.full(
        (random_resolution[1], random_resolution[0], 3), WHITE, dtype=np.uint8
    )
    number_of_lines = random.randint(1, MAX_NUMBER_OF_LINES)

    image_labels = []

    for j in range(number_of_lines):
        x1 = random.randint(0, random_resolution[0] - 1)
        x2 = random.randint(0, random_resolution[0] - 1)
        y1 = random.randint(0, random_resolution[1] - 1)
        y2 = random.randint(0, random_resolution[1] - 1)

        thickness = random.randint(1, 5)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)

        class_id, x_center, y_center, box_w, box_h = generate_data_for_labels(
            0, x1, x2, y1, y2, random_resolution
        )

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        image_labels.append(yolo_line)

    img = add_salt_and_pepper_noise(img, amount=0.01)
    base_filename = f"line{i}_{number_of_lines}"
    cv2.imwrite(f"daneTreningowe/images/{base_filename}.png", img)
    with open(f"daneTreningowe/labels/{base_filename}.txt", "w") as f:
        f.write("\n".join(image_labels))

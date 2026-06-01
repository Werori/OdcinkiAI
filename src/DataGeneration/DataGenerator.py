import os
import random

import cv2
import numpy as np


class DataGenerator:
    def __init__(self, root_dir: str = "Dataset", split_ratio: float = 0.8):
        self.root_dir = root_dir
        self.split_ratio = split_ratio
        self.resolution = (640, 480)
        self.white = 255
        self.train_images_path = os.path.join(root_dir, "TrainingData", "images")
        self.train_labels_path = os.path.join(root_dir, "TrainingData", "labels")

        self.val_images_path = os.path.join(root_dir, "ValidationData", "images")
        self.val_labels_path = os.path.join(root_dir, "ValidationData", "labels")
        os.makedirs(self.train_images_path, exist_ok=True)
        os.makedirs(self.train_labels_path, exist_ok=True)
        os.makedirs(self.val_images_path, exist_ok=True)
        os.makedirs(self.val_labels_path, exist_ok=True)

    def generate_data_for_labels(
        self,
        class_id: int,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        resolution: tuple[int, int],
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

    def _add_salt(self, image: np.ndarray, count: int) -> np.ndarray:
        height, width, _ = image.shape
        for _ in range(count):
            y = random.randint(0, height - 1)
            x = random.randint(0, width - 1)
            image[y, x] = [255, 255, 255]
        return image

    def _add_pepper(self, image: np.ndarray, count: int) -> np.ndarray:
        height, width, _ = image.shape
        for _ in range(count):
            y = random.randint(0, height - 1)
            x = random.randint(0, width - 1)
            image[y, x] = [0, 0, 0]
        return image

    def add_salt_and_pepper_noise(
        self, image: np.ndarray, amount: float = 0.01
    ) -> np.ndarray:
        noisy_image = image.copy()
        height, width, _ = image.shape
        num_pixels = height * width

        num_salt = int(np.ceil(amount * num_pixels * 0.5))
        num_pepper = int(np.ceil(amount * num_pixels * 0.5))

        noisy_image = self._add_salt(noisy_image, num_salt)
        noisy_image = self._add_pepper(noisy_image, num_pepper)
        return noisy_image

    def build_dataset(self, number_of_images: int, max_lines: int):
        for i in range(number_of_images):
            img = np.full(
                (self.resolution[1], self.resolution[0], 3),
                self.white,
                dtype=np.uint8,
            )
            number_of_lines = random.randint(1, max_lines)
            image_labels = []

            for _ in range(number_of_lines):
                x1 = random.randint(0, self.resolution[0] - 1)
                x2 = random.randint(0, self.resolution[0] - 1)
                y1 = random.randint(0, self.resolution[1] - 1)
                y2 = random.randint(0, self.resolution[1] - 1)

                thickness = random.randint(1, 5)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)

                class_id, x_center, y_center, box_w, box_h = (
                    self.generate_data_for_labels(0, x1, x2, y1, y2, random_resolution)
                )

                yolo_line = (
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                )
                image_labels.append(yolo_line)

            img = self.add_salt_and_pepper_noise(img, amount=0.01)
            base_filename = f"line{i}_{number_of_lines}"

            if random.random() < self.split_ratio:
                img_save_dir = self.train_images_path
                lbl_save_dir = self.train_labels_path
            else:
                img_save_dir = self.val_images_path
                lbl_save_dir = self.val_labels_path

            cv2.imwrite(os.path.join(img_save_dir, f"{base_filename}.png"), img)
            with open(os.path.join(lbl_save_dir, f"{base_filename}.txt"), "w") as f:
                f.write("\n".join(image_labels))

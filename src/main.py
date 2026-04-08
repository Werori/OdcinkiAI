import random

import cv2
import numpy as np

if __name__ == "__main__":
    NUMBER_OF_IMAGES = 1000
    WHITE = 255

    resolutions = [
            (640, 480),
            (320, 240),
            (160, 120)
            ]

    for i in range(0, NUMBER_OF_IMAGES):
        random_resolution = resolutions[random.randint(0,len(resolutions)-1)]

        img = np.full(( random_resolution[0], random_resolution[1], 3),WHITE, dtype=np.uint8)

        number_of_lines = random.randint(1, 100)

        for j in range(number_of_lines):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            x1 = random.randint(0, random_resolution[0])
            x2 = random.randint(0, random_resolution[0])
            y1 = random.randint(0, random_resolution[1])
            y2 = random.randint(0, random_resolution[1])

            thickness = random.randint(1, 5)
            
            cv2.line(img, (x1, y1), (x2, y2), (r, g, b), thickness)

        cv2.imwrite(f"images/line{i}.png", img)

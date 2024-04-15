import numpy as np
import math
import image_util
import os
import matplotlib.pyplot as plt

BASE_DIRECTORY = "/home/boris/.config/JetBrains/PyCharmCE2023.2/scratches/picturs/"
degree = 5  #angegbn in Grad æ ß , aber wir benötigen θ

def rotate_image_inverse_mapping(image, degree):
    #Convert degrees to radians
    angle_rad = np.radians(degree)

    cols = image.shape[0]
    rows = image.shape[1]
    # Calculate the center of the image
    center = (cols / 2, rows / 2)

    #initialize an empty rotated image with the original image size
    rotated_image = np.zeros_like(image, dtype=np.uint8)

    #calculate inverse transformation for each pixel in the rotated image
    for v in range(0, cols):
        for w in range(0, rows):
            #apply inverse rotation transformation and translation
            x = int((v - center[1]) * np.cos(angle_rad) - (w - center[0]) * np.sin(angle_rad) + center[1])
            y = int((v - center[1]) * np.sin(angle_rad) + (w - center[0]) * np.cos(angle_rad) + center[0])
                                # Matrizen multp wie war das nochmal ^                                                          ^ stimmt
            #interpolate pixel values using bilinear interpolation
            x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
            x_ceil, y_ceil = int(np.ceil(x)), int(np.ceil(y))

            if 0 <= x_floor < rows and 0 <= y_floor < cols and 0 <= x_ceil < rows and 0 <= y_ceil < cols:
                # Bilinear interpolation
                rotated_image[v, w] = (
                        (x_ceil - x) * (y_ceil - y) * image[y_floor, x_floor] +
                        (x - x_floor) * (y_ceil - y) * image[y_floor, x_ceil] +
                        (x_ceil - x) * (y - y_floor) * image[y_ceil, x_floor] +
                        (x - x_floor) * (y - y_floor) * image[y_ceil, x_ceil]
                )
            else:
                rotated_image[v, w] = 0  # Set to zero or handle out-of-bounds values accordingly
    # Translate the rotated image to align its center with the center of the original image
    translation = np.array([cols / 2, rows / 2]) - np.array([center[0], center[1]])
    rotated_image = np.roll(rotated_image, (int(translation[0]), int(translation[1])), axis=(0, 1))

    return rotated_image

#Load the image using imageio
image = image_util.read(os.path.join(BASE_DIRECTORY, 's_tannennadel.png'), as_gray=True)

#Rotate the image using inverse mapping with size adjustment
rotated_image = rotate_image_inverse_mapping(image, degree)

#Display the original and rotated images using matplotlib
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rotated Image")
plt.imshow(rotated_image, cmap='gray')
plt.axis("off")

plt.show()

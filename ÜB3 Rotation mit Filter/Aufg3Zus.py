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

    #get image shape
    rows = image.shape[0]
    cols = image.shape[1]

    #calculate the center of the image
    center = (cols / 2, rows / 2)

    #calculate the corners of the rotated bounding box
    corners = np.array([
        [0, 0],
        [cols, 0],
        [cols, rows],
        [0, rows]
    ])
    # corners = np.array([
    #     [-cols / 2, -rows / 2],
    #     [cols / 2, -rows / 2],
    #     [cols / 2, rows / 2],
    #     [-cols / 2, rows / 2]
    # ])

    #rotate the corners using the inverse rotation transformation
    # rotated_corners = np.dot(corners, np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
    #                                             [np.sin(angle_rad), np.cos(angle_rad)]]))
    rotated_top_left = np.dot([0,0], np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                                [np.sin(angle_rad), np.cos(angle_rad)]]))
    rotated_top_right = np.dot([cols,0], np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                                [np.sin(angle_rad), np.cos(angle_rad)]]))
    rotated_bottom_right = np.dot([cols, rows], np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                                [np.sin(angle_rad), np.cos(angle_rad)]]))
    rotated_bottom_left  = np.dot([0, rows], np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                                [np.sin(angle_rad), np.cos(angle_rad)]]))
    #find the minimum and maximum coordinates of the rotated corners
    min_x = int(np.min(rotated_top_left))
    min_y = int(np.min(rotated_top_right))
    max_x = int(np.max(rotated_bottom_left))
    max_y = int(np.max(rotated_bottom_right))

    #find the minimum and maximum coordinates of the rotated corners
    # min_x, min_y = int(np.min(rotated_corners))
    # max_x, max_y = int(np.max(rotated_corners))

    #calculate the size of the output image
    output_cols = int(np.ceil(max_x - min_x))
    output_rows = int(np.ceil(max_y - min_y))

    #initialize an empty rotated image with the original image size
    rotated_image = np.zeros_like(image, dtype=np.uint8)

    #calculate the translation to align the centre of rotated image with the center o fthe orinigal image
    translation = np.array([cols / 2, rows / 2]) + np.array([min_x, min_y])

    #calculate inverse transformation for each pixel in the rotated image
    for i in range(output_rows):
        for j in range(output_cols):
            # Apply inverse rotation transformation and translation
            x = int((i - center[1]) * np.cos(angle_rad) - translation[1] - (j - center[0]) * np.sin(angle_rad) + center[1] + translation[1])
            y = int((i - center[1]) * np.sin(angle_rad) + translation[1] + (j - center[0]) * np.cos(angle_rad) + center[0] + translation[0])
                                # Matrizen multp wie war das nochmal ^                                                          ^ stimmt
            # Check if the new coordinates are within bounds
            if 0 <= x < rows and 0 <= y < cols:
                rotated_image[x, y] = image[i, j]

    return rotated_image

# Load the image using imageio
image = image_util.read(os.path.join(BASE_DIRECTORY, 's_tannennadel.png'), as_gray=True)

# Rotate the image using inverse mapping with size adjustment
rotated_image = rotate_image_inverse_mapping(image, degree)

# Display the original and rotated images using matplotlib
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rotated Image (Geschrumpft auf Original)")
plt.imshow(rotated_image, cmap='gray')
plt.axis("off")

plt.show()

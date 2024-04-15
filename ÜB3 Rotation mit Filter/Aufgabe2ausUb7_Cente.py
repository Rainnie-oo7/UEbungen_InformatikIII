import numpy as np
import math
import image_util
import os
import matplotlib.pyplot as plt

BASE_DIRECTORY = "/home/boris/.config/JetBrains/PyCharmCE2023.2/scratches/picturs/"
degree = 5  #angegbn in Grad æ ß , aber wir benötigen θ

def rotate_forward(image, degree):
    #create an ,,empty´´ picture with the size of Unserem Bild
    output_image = np.zeros_like(image, dtype=np.uint8)

    #Umformng Grad in Radiants
    degreerad = np.radians(degree)
    #cols, rows = image.shape
    #Get image shape #Für wenn Rotat. nicht im Urspung sei sondern in der Mitte des Bilts
    cols  = image.shape[0]
    rows  = image.shape[1]
    #Calculate the center of the image
    center = (cols / 2, rows / 2)

    M = image.shape[0]
    N = image.shape[1]
    for v in range(0, M):
        for w in range(0, N):
            #FALSCH-> #output_image[v, w] = image[int(v * math.cos(degree) + v * math.sin(degree)), int(w * (-math.sin(degree)) + w * math.cos(degree))]
            # Apply rotation transformation
            # x = int( v * np.cos(degreerad) - w * np.sin(degreerad) )
            # y = int( v * np.sin(degreerad) + w * np.cos(degreerad) )
            # Für wenn Rotat. nicht im Urspung sei sondern in der Mitte des Bilts
            x = int((v - center[1]) * np.cos(degreerad) - (v - center[0]) * np.sin(degreerad) + center[1])
            y = int((w - center[1]) * np.sin(degreerad) + (w - center[0]) * np.cos(degreerad) + center[0])

            # Check if the new coordinates are within bounds
            # if 0 <= x < rows and 0 <= y < cols:
            #    output_image[v, w] = image[x, y]

            #Check if the new coordinates are within bounds
            if 0 <= x < M and 0 <= y < N:
                output_image[x, y] = image[v, w]

    return output_image

#Load image
image = image_util.read(os.path.join(BASE_DIRECTORY, 's_tannennadel.png'), as_gray=True)

#Load image, # Apply the filter
output_image = rotate_forward(image, degree)
#
#rotate_forward(image, degree)

# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Rotation-Applied Image')

plt.show()

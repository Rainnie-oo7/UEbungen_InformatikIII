import os
import image_util
from matplotlib import pyplot as plt
import numpy as np

BASE_DIRECTORY = '/home/boris/.config/JetBrains/PyCharmCE2023.3/InformatikIII/Übungen-InformatikIII/picturs'

import numpy as np
import os
import imageio
import image_util
from skimage import io, color
from scipy.linalg import solve
import matplotlib.pyplot as plt

BASE_DIRECTORY = '/home/boris/.config/JetBrains/PyCharmCE2023.3/InformatikIII/Übungen-InformatikIII/picturs'


# Benutzer festgelegte Intensität
# user_intensity = 50

# read image
def read(path, as_gray=False):
    image = io.imread(path)
    if as_gray and len(image.shape) > 2:
        image = (255 * color.rgb2gray(image)).astype(np.uint8)  # convert to grayscale
    return image


# write image
def write(path, image):
    io.imsave(path, image)


# show image using matplotlib
def show(image, dpi=100):
    # define plot
    _, ax = plt.subplots(dpi=dpi)

    # title
    plt.title('Image Fusion')

    # set label position
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xlabel('y-axis', fontsize=16)
    plt.ylabel('x-axis', fontsize=16)
    # show image
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='none')
    plt.show()

def find_min_intensity_coordinates(image):
    list = []

    # Bildgröße
    height, width = image.shape

    # Teile das Bild in vier Quadranten
    quad1 = image[:height // 2, :width // 2]
    quad2 = image[:height // 2, width // 2:]
    quad3 = image[height // 2:, :width // 2]
    quad4 = image[height // 2:, width // 2:]

    # Get the height and width
    height, width, _ = quad1.shape

    # Append the new image dimensions to the list
    list.append({"height": height, "width": width})

    height, width, _ = quad2.shape
    list.append({"height": height, "width": width})

    height, width, _ = quad3.shape
    list.append({"height": height, "width": width})

    height, width, _ = quad4.shape
    list.append({"height": height, "width": width})


def calculate_coefficients(quad1, quad2, quad3, quad4):
    # Konvertiere das Bild zu einer eindimensionalen Matrix
    for height in list:
        for width in list:
            for x in range(0, height):  # to sum x coordinates
                for y in range(0, width):
                    b = image[x, y]  # intensity at point x,y

    # Berechne die Koeffizienten a0, a1, a2 mithilfe von solve
    coefficients = solve(A.T @ A, A.T @ b)

    return coefficients

    # Funktion zur Berechnung der minimalen Intensität und ihrer Koordinaten


def min_intensity_coordinates(quad):
    min_intensity = np.min(quad)
    min_index = np.argmin(quad)
    min_y, min_x = np.unravel_index(min_index, quad.shape)
    return min_intensity, min_x, min_y

    # Berechne die minimalen Intensitäten und ihre Koordinaten für jeden Quadranten
    min_intensity_1, min_x_1, min_y_1 = min_intensity_coordinates(quad1)
    min_intensity_2, min_x_2, min_y_2 = min_intensity_coordinates(quad2)
    min_intensity_3, min_x_3, min_y_3 = min_intensity_coordinates(quad3)
    min_intensity_4, min_x_4, min_y_4 = min_intensity_coordinates(quad4)

    return (min_intensity_1, min_x_1, min_y_1), (min_intensity_2, min_x_2, min_y_2), \
        (min_intensity_3, min_x_3, min_y_3), (min_intensity_4, min_x_4, min_y_4)


def generate_image(intensity, coefficients):
    # Bildgröße
    width, height = 512, 512

    # Erstelle ein Gitter von x- und y-Werten
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)

    # Erstelle ein 2D-Gitter
    x, y = np.meshgrid(x, y)

    # Parameter für die Funktion I(x, y)
    a0, a1, a2 = 0.5, 0.3, 0.2

    # Calc. Intensitätswerte basierend auf der Funktion
    intensity_values = a0 + a1 * x + a2 * y

    # Skalierung der Intensität auf den vom Benutzer gewählten Wert
    intensity_values *= intensity

    # Konvertiere, für Graustufen
    intensity_values = (intensity_values * 255).astype(np.uint8)

    # Erstelle das Bild
    generated_image = np.stack([intensity_values] * 3, axis=-1)

    return generated_image


# Save image into local hard drive

def subtract_background(background_path, foreground_path):
    # Lade Bilder
    foreground_path = image_util.read(os.path.join(BASE_DIRECTORY, 'fluo_shading.png'), as_gray=True)

    # Überprüfe, ob die Dimensionen der Bilder übereinstimmen
    if generated_image.shape != foreground_path.shape:
        raise ValueError("Die Dimensionen der Bilder stimmen nicht überein.")

    # Höhe und Breite der Bilder
    height, width, _ = generated_image.shape

    # Initialisiere leeres Bild für das Ergebnis
    result = np.zeros_like(generated_image)

    # Iteriere über jedes Pixel und subtrahiere den Hintergrund vom Vordergrund #hier shape[0]/2 #dann argmin??? (Suche nach mijimalen Intensitätswert
    for y in range(height):
        for x in range(width):
            result[y, x] = foreground_path[y, x] - generated_image[y, x]

    return result


# # Pfade zu den Bildern
# foreground_path = image_util.read(os.path.join(BASE_DIRECTORY, 'fluo_shading.png'), as_gray=True)
# # background_path = "hintergrundbild.jpg"
#
# # Berechne die Koeffizienten a0, a1, a2
# coefficients = calculate_coefficients(generated_image)
#
# # Gebe die Koeffizienten aus
# print("Koeffizienten (a0, a1, a2):", coefficients)
#
# # Finde die minimalen Intensitäten und ihre Koordinaten für jeden Quadranten
# min_coords_quad1, min_coords_quad2, min_coords_quad3, min_coords_quad4 = find_min_intensity_coordinates(example_image)
#
# # Drucke die Ergebnisse
# print("Quadrant 1: Min Intensität und Koordinaten:", min_coords_quad1)
# print("Quadrant 2: Min Intensität und Koordinaten:", min_coords_quad2)
# print("Quadrant 3: Min Intensität und Koordinaten:", min_coords_quad3)
# print("Quadrant 4: Min Intensität und Koordinaten:", min_coords_quad4)
#
# # Generiere das Bild mit der gewünschten Intensität
# generated_image = generate_image(intensity)
# # generated_image = generate_image(user_intensity)
#
# # Subtrahiere den Hintergrund vom Vordergrund
# result_image = subtract_background(generated_image, foreground_path)
# # (background_path, foreground_path)
#
#
# # Zeige das Bild an
# plt.imshow(generated_image, cmap='gray')
# plt.axis('off')
# plt.show()
#
# # Zeige das Ergebnisbild an
# plt.imshow(result_image)
# plt.axis('off')
# plt.show()

if __name__ == '__main__':

    image = image_util.read( os.path.join(BASE_DIRECTORY, 'fluo_shading.png'))

    image_util.show(image) #show image with background

    M = image.shape[0]
    N = image.shape[1]

    import numpy as np
    # Coefficients of the equations
    coefficients = np.array([
        [1, 26, 18],
        [1, 136, 236],
        [1, 493, 496],
    ])

    # Constants on the right-hand side of the equations
    constants = np.array([15, 52, 126])

    # Solve the system of equations
    solution = np.linalg.solve(coefficients, constants)
    variables = (np.floor(solution)).astype(int)  #solution.astype(int) #np.round(solution, decimals=0)

    print("Solution for  coefficients a0, a1, a2:")
    print(solution)
    print(variables)

    background_image = []

    for x in range(0, M):
        rows = []
        for y in range(0, N):
            background_intensity = solution[0] + (solution[1]*x) + (solution[2]*y)
            rows.append(background_intensity)
            image[x,y] = abs(image[x,y] - background_intensity)
        background_image.append(rows)

    image_util.show(background_image)


    image_util.show(image) #image without background

    # min intenstity in first square
    square_1 = [] # contains all intensities in that square
    for x in range (0, int(M/2)):
        for y in range (0, int(N/2)):
            square_1.append([image[x,y], (x,y)])

    print(min(square_1))

    square_2 = []  # contains all intensities in that square
    for x in range(0, int(M / 2)):
        for y in range(int(N / 2), N):
            square_2.append([image[x, y],(x,y)])

    print(min(square_2))

    square_3 = []  # contains all intensities in that square
    for x in range(int(M / 2), M):
        for y in range(0, int(N / 2)):
            square_3.append([image[x, y], (x, y)])

    print(min(square_3))

    square_4 = []  # contains all intensities in that square
    for x in range(int(M / 2), M):
        for y in range(int(N / 2), N):
            square_4.append([image[x, y], (x, y)])

    print(min(square_4))
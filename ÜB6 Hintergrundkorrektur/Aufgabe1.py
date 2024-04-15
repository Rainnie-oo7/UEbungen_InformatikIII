import os
import image_util
from matplotlib import pyplot as plt
import numpy as np

BASE_DIRECTORY = '/home/boris/.config/JetBrains/PyCharmCE2023.2/scratches/picturs/'

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
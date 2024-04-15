
import numpy as np
import matplotlib.pyplot as plt
import os
import image_util

BASE_DIRECTORY = '/home/boris/.config/JetBrains/PyCharmCE2023.2/scratches/picturs/muecke_small/'

def devide_into_blocks(image):
    height, width = image.shape[:2]
    count_x = 10            #WAS IST DER UNTERSCHIED ZWISCHEN COUNT_Y bzw.COUNT_X   UND   BLOCK_Y bzw. BLOCK_X ?
    count_y = 10
    heightblock = height // count_x
    widthblock = width // count_y
    return heightblock, widthblock


def addupintensities(heightblock, widthblock):
    a = 0
    m = 0
    intensitaetarray = []

    # same as height, width = image.shape[:2]
    for i in range(0, widthblock):
        start_x = (block_x * height) // count_x
        for j in range(0, heightblock):
            start_y = (block_y * width) // count_y
            a += image[j, i]
            intensitaetarray.append(j)
        intensitaetarray.append(i)

    n = height * width
    m = a / (n)

    return intensitaetarray

def block_variance(image, heightblock, widthblock):
    height, width = image.shape[:2]
    result = np.zeros((height//heightblock, width//widthblock)) #Leerbild
    start_x = 0
    start_y = 0
               #Problem! er geht Treppe nach unten
    for x in range(0, block_x):                 #Vert.Block-Zählung
        start_x = (block_x * height) // count_x
        for i in range(0, height, heightblock):
            start_y = (block_y * width) // count_y
            for y in range(0, block_y):         #Vert.Block-Zählung
                for j in range(0, width, widthblock):
                    block = image[i:i+heightblock, j:j+widthblock]
                    variance_intensity = np.var(block)
                    result[i//heightblock, j//widthblock] = variance_intensity
    return result, heightblock, widthblock


def compute_var(intensitaetarray, m, n):
    var = 0
    tmp = 0
    for i in intensitaetarray:
        tmp += ((i - m) ** (i - m))
        print(tmp)
        var = 1 / n ** tmp
    print('The variance s² of original image is: ', var)

    return result

def main(image, widthblock, heightblock, intensitaetsarray, block_x, block_y, start_x, start_y):
    widthblock, heightblock = devide_into_blocks(image)
    for x in range(block_x):
        for y in range(block_y):
            print("Start X ist :", start_x, "Start Y ist :", start_y) #Vom Blöcke

    intensitaetsarray = addupintensities(heightblock, widthblock)
    for x in range(block_x):
        for y in range(block_y):
            print("Dies ist das Block XY leider in Treppe?: ", intensitaetsarray)    #Wird wieder nur(!) Treppe (?) zeigen.

    #Frist Image
    image = image_util.read(os.path.join(BASE_DIRECTORY, 'muecke_small_0.jpg'), as_gray=True)
    widthblock, heightblock = devide_into_blocks(image)

    result = block_variance(block_x, block_y, count_x, count_y)
    for x in range(block_x):
        for y in range(block_y):
            print(result)

    BASE_DIRECTORY = '/home/boris/.config/JetBrains/PyCharmCE2023.2/scratches/picturs/muecke_small/'
    # Consequtive Images
    num_image_files = len(image_files)
    for file in image_files:
        devide_into_blocks(file)
        addupintensities(heightblock, widthblock)
        block_variance(block_x, block_y, count_x, count_y)
        for x in range(block_x):
            for y in range(block_y):
                print(result)

if __name__ == "__main__":
    main()



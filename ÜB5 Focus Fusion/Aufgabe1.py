import numpy as np
import imageio
import image_util
import os
import matplotlib.pyplot as plt

BASE_DIRECTORY = "/home/boris/.config/JetBrains/PyCharmCE2023.2/scratches/picturs/muecke_small"

def read_image(BASE_DIRECTORY, as_gray=False):
    #nope v
    #images = [imageio.imread(ein_bild) for ein_bild in file_paths]
    #image = [image_util.read(os.path.join(BASE_DIRECTORY, file), as_gray=True) for file in BASE_DIRECTORY]
    image = image_util.read(os.path.join(BASE_DIRECTORY, 'muecke_small_0.jpg'), as_gray=True)

    return image

def calculate_block_variances(image_files, block_size = 10):
    a   = 0
    m   = 0
    var = 0
    tmp = 0
    first_image = read_image(os.path.join(BASE_DIRECTORY, image_files[0]))
    image_file_dimensions = first_image.shape

    M = image_file_dimensions[0]  # height
    N = image_file_dimensions[1]  # width
    variances = np.zeros((M//block_size, N//block_size))
    for file in image_files:
        image = read_image(os.path.join(BASE_DIRECTORY, file), as_gray=True)
        for i in range(0, M, block_size):
            for j in range(0, N, block_size):
                height = file.shape[0]  # Y
                width = file.shape[1]
                block = image[i:i+block_size, j:j+block_size]
                for k in block:
                    for i in range(0, width):  # Image has no "height" bei image.height \ height.image #daher von 0 bis height
                        for j in range(0, height):  # Achtung auf den Einschub! Alte Schleife beginnt ->| ... !!!
                            a += image[j, i]
                            block.append(j)
                        block.append(i)
                    n = height * width
                    m = a / (n)                # Mean
                    tmp += ((k - m) ** (k - m))
                    print('Der Vor-Bruch (Vor-Bemittelrationalisierung) tmp ist :' , tmp)
                    var = 1 / n ** tmp
                print('The variance s² of jedem Block ist: ', var)
                variances[i//block_size, j//block_size] = var       # Variance hier Array gemeint
                #variances[i//block_size, j//block_size] = np.var(block)

    return var, variances

def fuse_images(image_files, block_size):
    num_images, height, width = image_files.shape
    fused_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block_variances = [calculate_block_variances(image[i:i+block_size, j:j+block_size], block_size) for image in image_files]
            min_variance_index = np.argmin(block_variances[0] + block_variances[1] + block_variances[2])
            fused_image[i:i+block_size, j:j+block_size] = image_files[min_variance_index][i:i+block_size, j:j+block_size]

    return fused_image

def main():
    block_size = 10

    #nope v
    #file_paths = ['muecke_small_0', 'muecke_small_1', 'muecke_small_2', 'muecke_small_3', 'muecke_small_4', 'muecke_small_5', 'muecke_small_6', 'muecke_small_7', 'muecke_small_8', 'muecke_small_9']
    n = 10  # n describes num of files to choose from
    image_files = []
    for i in range(n):
        name = 'muecke_small_' + str(i) + '.jpg'
        image_files.append(name)
    #nope v
    #images = np.stack(read_images(BASE_DIRECTORY), axis=0)
    #images = np.stack(read_images(file_paths), axis=0)


    # Startbild (Breite Höhe Werte)
    variances = calculate_block_variances(image_files, block_size = 10)
    var = calculate_block_variances(image_files, block_size = 10)
    print('vars: ', var)
    # print('Variances: ', variances)

    fused_image = fuse_images(image_files, block_size)

    plt.imshow(fused_image, cmap='gray')
    plt.title('Focus Fusion')
    plt.show()

if __name__ == "__main__":
    main()

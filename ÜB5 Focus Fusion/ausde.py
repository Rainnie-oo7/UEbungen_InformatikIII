import numpy as np
import imageio
import matplotlib.pyplot as plt

def read_images(file_paths):
    images = [imageio.imread(file_path) for file_path in file_paths]
    return images

def calculate_block_variances(image, block_size):
    height, width = image.shape
    variances = np.zeros((height//block_size, width//block_size))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            variances[i//block_size, j//block_size] = np.var(block)

    return variances

def fuse_images(images, block_size):
    num_images, height, width = images.shape
    fused_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block_variances = [calculate_block_variances(image[i:i+block_size, j:j+block_size], block_size) for image in images]
            min_variance_index = np.argmin(block_variances[0] + block_variances[1] + block_variances[2])
            fused_image[i:i+block_size, j:j+block_size] = images[min_variance_index][i:i+block_size, j:j+block_size]

    return fused_image

def main():
    file_paths = ['mucke_small_0.jpg', 'mucke_small_1.jpg', 'mucke_small_2.jpg']  # replace with your image file paths
    images = np.stack(read_images(file_paths), axis=0)
    block_size = 16

    fused_image = fuse_images(images, block_size)

    plt.imshow(fused_image, cmap='gray')
    plt.title('Focus Fusion')
    plt.show()

if __name__ == "__main__":
    main()

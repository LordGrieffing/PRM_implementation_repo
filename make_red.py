import cv2
import numpy as np

def change_non_black_white_to_red(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)


    imgHeight, imgWidth, channels = img.shape


    for i in range(imgHeight):
        for j in range(imgWidth):
            color = img[i, j]
            if not np.all(color == 0) and not np.all(color == 255):
                img[i, j] = [0, 0, 255]
    # Save the result
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    # Path to the input image and output image
    input_path = "map_sequences/skeleton_on_map_5.png"
    output_path = "map_sequences/skeleton_on_map_5.png"

    change_non_black_white_to_red(input_path, output_path)
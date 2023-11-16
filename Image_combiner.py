import cv2
import numpy as np

def copy_non_black_pixels(source_path, destination_path, output_path):
    # Read the source and destination images
    source_image = cv2.imread(source_path)
    destination_image = cv2.imread(destination_path)

    # Ensure the images have the same size
    if source_image.shape != destination_image.shape:
        raise ValueError("Images must have the same size")

    # Create a mask for non-black pixels in the source image
    non_black_mask = np.any(source_image != [0, 0, 0], axis=-1)

    # Copy non-black pixels from the source image to the destination image
    result = np.copy(destination_image)
    result[non_black_mask] = source_image[non_black_mask]

    # Save the result
    cv2.imwrite(output_path, result)

if __name__ == "__main__":
    # Paths to the source, destination, and output images
    source_path = "map_sequences/skeleton_map_sequence_5.png"
    destination_path = "map_sequences/map_sequence_5.png"
    output_path = "map_sequences/skeleton_on_map_5.png"

    copy_non_black_pixels(source_path, destination_path, output_path)
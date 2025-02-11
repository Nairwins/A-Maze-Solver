from PIL import Image
import numpy as np
# Open the image
image = Image.open('../Test/maze.png').convert('L')  # Convert to grayscale

# Convert the image to a numpy array
image_array = np.array(image)

# Create a binary matrix: 1 for black pixels, 0 for the rest
binary_matrix = np.where(image_array == 0, 1, 0)

# Print the binary matrix
print(binary_matrix)

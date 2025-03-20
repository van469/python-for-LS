import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path = r'/res/img_1.tif'
image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# No need to convert to grayscale again


binary_image1 = cv2.threshold(image1, 185, 255, cv2.THRESH_BINARY)[1]


# Convert the binary image to RGB for matplotlib display
image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
binary_image_rgb = cv2.cvtColor(binary_image1, cv2.COLOR_GRAY2RGB)


# Find contours in the binary image
contours, _ = cv2.findContours(binary_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Count the number of contours
num_colonies = len(contours)
print(f"Number of cells in {image_path}: {num_colonies}")


# Draw contours on the original image for visualization
image_with_contours = cv2.drawContours(image_rgb.copy(), contours, -1, (0, 150, 0), 2)


    # Display the original and binary images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image with Contours")
plt.imshow(image_with_contours)
plt.axis('off')


plt.subplot(1, 2, 2)
plt.title("Binary Image")
plt.imshow(binary_image_rgb)
plt.axis('off')


plt.show()




FOR IMAGE 1, THE NUMBER OF CELLS ARE 173




#Image 2


image_path2 = r'/res/img_2.tif'
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)


# No need to convert to grayscale again


binary_image2 = cv2.threshold(image2, 185, 255, cv2.THRESH_BINARY)[1]


# Convert the binary image to RGB for matplotlib display
image_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
binary_image_rgb = cv2.cvtColor(binary_image2, cv2.COLOR_GRAY2RGB)


# Find contours in the binary image
contours, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Count the number of contours
num_colonies = len(contours)
print(f"Number of cells in {image_path2}: {num_colonies}")


# Draw contours on the original image for visualization
image_with_contours = cv2.drawContours(image_rgb.copy(), contours, -1, (0, 255, 0), 2)


    # Display the original and binary images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image with Contours")
plt.imshow(image_with_contours)
plt.axis('off')


plt.subplot(1, 2, 2)
plt.title("Binary Image")
plt.imshow(binary_image_rgb)
plt.axis('off')


plt.show()


FOR IMAGE 2. THE NUMBER OF CELLS ARE 172

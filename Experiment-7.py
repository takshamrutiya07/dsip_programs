import cv2
import numpy as np
import matplotlib.pyplot as plt
src_image = cv2.imread('BMW.jpg', cv2.IMREAD_GRAYSCALE)
if src_image is None:
    print("Image not found!")
    exit()
negative_image = 255 - src_image
_, thresholded_image = cv2.threshold(src_image, 128, 255, cv2.THRESH_BINARY)
gamma = 2.0
normalized_image = src_image / 255.0
gamma_corrected_image = np.power(normalized_image, 1/gamma)
gamma_corrected_image = np.uint8(gamma_corrected_image * 255)
plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.imshow(src_image, cmap='gray')
plt.title("Original")
plt.axis('off')
plt.subplot(1,4,2)
plt.imshow(negative_image, cmap='gray')
plt.title("Negative")
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Threshold")
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(gamma_corrected_image, cmap='gray')
plt.title("Gamma")
plt.axis('off')
plt.show()

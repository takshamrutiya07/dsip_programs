import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("firstimages.jpg")

if image is None:
    print("Error: Image not found")
    exit()
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 5))
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 5))
plt.imshow(gray_image, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()


print("Minimum intensity:", np.min(gray_image))
print("Maximum intensity:", np.max(gray_image))



r_min = np.min(gray_image)
r_max = np.max(gray_image)

stretched = ((gray_image - r_min) / (r_max - r_min)) * 255
stretched = stretched.astype(np.uint8)

plt.figure(figsize=(8, 5))
plt.imshow(stretched, cmap="gray")
plt.title("Enhanced Image - Contrast Stretching")
plt.axis("off")
plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Before Contrast Stretching")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(stretched, cmap="gray")
plt.title("After Contrast Stretching")
plt.axis("off")

plt.show()



image = cv2.imread("my_img.jpg")

if image is None:
    print("Error: Image not found")
    exit()

print("Image loaded successfully!")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(rgb_image)
plt.title("Original")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
gamma = 0.5
normalized = gray_image / 255.0
gamma_image = np.power(normalized, gamma)

gamma_image = np.uint8(gamma_image * 255)

plt.figure(figsize=(8, 6))
plt.imshow(gamma_image, cmap="gray")
plt.title("Enhanced Image - Gamma Transformation")
plt.axis("off")
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Before Gamma Transformation")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gamma_image, cmap="gray")
plt.title("After Gamma Transformation")
plt.axis("off")

plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(gray_image.ravel(), bins=256, range=[0, 256])
plt.title("Histogram Before Enhancement")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.subplot(1, 2, 2)
plt.hist(gamma_image.ravel(), bins=256, range=[0, 256])
plt.title("Histogram After Gamma Transformation")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.show()

print("Before Enhancement:")
print("Minimum intensity:", np.min(gray_image))
print("Maximum intensity:", np.max(gray_image))

print("\nAfter Gamma Transformation:")
print("Minimum intensity:", np.min(gamma_image))
print("Maximum intensity:", np.max(gamma_image))

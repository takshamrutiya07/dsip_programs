import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("secondimage.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel_size = (5, 5)
sigma = 1.5

gaussian_1d = cv2.getGaussianKernel(kernel_size[0], sigma)
gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)
smoothed_rgb = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)

sharpening_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
sharpened_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(smoothed_rgb)
plt.title("Gaussian Smoothed Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(sharpened_rgb)
plt.title("Sharpened Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gaussian_kernel, cmap="gray")
plt.title("Gaussian Kernel")
plt.axis("off")

plt.tight_layout()
plt.show()




import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("secondimage.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel_size = (5, 5)

kernel = np.ones(kernel_size, dtype=np.float32)
kernel = kernel / (kernel_size[0] * kernel_size[1])

smoothed_image = cv2.filter2D(image, -1, kernel)
smoothed_rgb = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(smoothed_rgb)
plt.title("Averaging Filter Smoothed Image")
plt.axis("off")

plt.tight_layout()
plt.show()




import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("secondimage.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel_size = 5

smoothed_image = cv2.medianBlur(image, kernel_size)
smoothed_rgb = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(smoothed_rgb)
plt.title("Median Filter Smoothed Image")
plt.axis("off")

plt.tight_layout()
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("secondimage.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
blurred_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

high_pass_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
], dtype=np.float32)

sharpened_image = cv2.filter2D(
    blurred_image, -1, high_pass_kernel
)
sharpened_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred_rgb)
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened_rgb)
plt.title("High-Pass Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()
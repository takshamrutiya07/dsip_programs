import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_median_filter(image, kernel_size):
# Apply median filter to remove noise
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


def apply_bilateral_filter(image, d, sigma_color, sigma_space):
# Apply bilateral filter to remove noise
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image
# Load the input image


image_path = 'T.jpg'
input_image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Apply median filter
median_filtered_image = apply_median_filter(image_rgb, kernel_size=5)
plt.imshow(image_rgb)
plt.show()
plt.imshow(median_filtered_image)
plt.show()
bilateral_filtered_image = apply_bilateral_filter(image_rgb, d=9, sigma_color=75,
sigma_space=75)
plt.imshow(bilateral_filtered_image)
plt.show()

from PIL import Image, ImageFilter
# Open an image
img = Image.open('T.jpg')

# Apply a max filter with a kernel size of 3
max_filtered = img.filter(ImageFilter.MaxFilter(size=3))
plt.imshow(max_filtered)
plt.show()

# Apply a mean filter (box blur with radius 1 creates a 3x3 mean average)
min_filtered = img.filter(ImageFilter.MinFilter(size=3))
plt.imshow(min_filtered)
plt.show()




import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Open the image in grayscale
img = cv2.imread('T.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Could not find or open 'T.jpg'")

# 2. Define a 3x3 structuring element (kernel)
kernel = np.ones((9, 9 ), np.uint8)

# 3. Apply the filters
max_filtered = cv2.dilate(img, kernel)  # Max Filter
min_filtered = cv2.erode(img, kernel)   # Min Filter

# 4. Plot the Images side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Max Filter (Dilation)
axes[1].imshow(max_filtered, cmap='gray')
axes[1].set_title('Max Filter (Dilate)')
axes[1].axis('off')

# Min Filter (Erosion)
axes[2].imshow(min_filtered, cmap='gray')
axes[2].set_title('Min Filter (Erode)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

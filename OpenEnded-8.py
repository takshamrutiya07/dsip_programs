import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(source, reference):

    # Get histogram of source image
    source_hist, _ = np.histogram(
        source.flatten(),
        bins=256,
        range=(0, 256),
        density=True
    )

    # Get histogram of reference image
    reference_hist, _ = np.histogram(
        reference.flatten(),
        bins=256,
        range=(0, 256),
        density=True
    )

    # Calculate cumulative distribution function
    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)

    # Normalize CDF
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        difference = np.abs(reference_cdf - source_cdf[i])

        lookup_table[i] = np.argmin(difference)

    # Apply lookup table
    matched_image = cv2.LUT(source, lookup_table)

    return matched_image


	 
image1 = cv2.imread(r"D:\Semester-5\DSIP\Lab_Manual\firstimages.jpg")
image2 = cv2.imread(r"D:\Semester-5\DSIP\Lab_Manual\secondimage.jpg")

if image1 is None or image2 is None:
    print("Error: One or more images not found")
    exit()

print("Images loaded successfully!")
rgb_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
rgb_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


equalized_image1 = cv2.equalizeHist(gray_image1)

equalized_image2 = cv2.equalizeHist(gray_image2)


matched_image1 = histogram_matching(
    gray_image1,
    gray_image2
)

matched_image2 = histogram_matching(
    gray_image2,
    gray_image1
)


plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(rgb_image1)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(gray_image1, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(equalized_image1, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(matched_image1, cmap="gray")
plt.title("Histogram Matching")
plt.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(rgb_image2)
plt.title("Original second")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(gray_image2, cmap="gray")
plt.title("Grayscale second")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(equalized_image2, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(matched_image2, cmap="gray")
plt.title("Histogram Matching")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(
    gray_image1.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Original Histogram - first")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.subplot(1, 3, 2)
plt.hist(
    equalized_image1.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Equalized Histogram - first")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.subplot(1, 3, 3)
plt.hist(
    matched_image1.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Matched Histogram - first")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(
    gray_image2.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Original Histogram - second")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.subplot(1, 3, 2)
plt.hist(
    equalized_image2.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Equalized Histogram - second")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.subplot(1, 3, 3)
plt.hist(
    matched_image2.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Matched Histogram - second")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()

print("\nOriginal first:")
print("Minimum Intensity:", np.min(gray_image1))
print("Maximum Intensity:", np.max(gray_image1))

print("\nAfter Histogram Equalization:")
print("Minimum Intensity:", np.min(equalized_image1))
print("Maximum Intensity:", np.max(equalized_image1))

print("\nAfter Histogram Matching:")
print("Minimum Intensity:", np.min(matched_image1))
print("Maximum Intensity:", np.max(matched_image1))

print("\nOriginal second:")
print("Minimum Intensity:", np.min(gray_image2))
print("Maximum Intensity:", np.max(gray_image2))

print("\nAfter Histogram Equalization:")
print("Minimum Intensity:", np.min(equalized_image2))
print("Maximum Intensity:", np.max(equalized_image2))

print("\nAfter Histogram Matching:")
print("Minimum Intensity:", np.min(matched_image2))
print("Maximum Intensity:", np.max(matched_image2))

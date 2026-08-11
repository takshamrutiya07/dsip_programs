import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

original, sr = librosa.load("original.mp3", sr=22050, mono=True)
karaoke, sr = librosa.load("original_k.mp3", sr=22050, mono=True)
different, sr = librosa.load("different.mp3", sr=22050, mono=True)

n = min(len(original), len(karaoke), len(different))

original = original[:n]
karaoke = karaoke[:n]
different = different[:n]

print("Files Loaded Successfully")
print("Number of Samples:", n)

c1 = np.corrcoef(original, karaoke)[0, 1]
c2 = np.corrcoef(original, different)[0, 1]
c3 = np.corrcoef(karaoke, different)[0, 1]

print("\nCorrelation Results")
print("-------------------")
print("Original vs Karaoke  :", round(c1, 4))
print("Original vs Different:", round(c2, 4))
print("Karaoke vs Different :", round(c3, 4))

matrix = np.array([
    [1, c1, c2],
    [c1, 1, c3],
    [c2, c3, 1]
])

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(original, color="blue")
plt.title("Original Song")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(karaoke, color="green")
plt.title("Karaoke Version")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(different, color="red")
plt.title("Different Song")
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))

plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    [0, 1, 2],
    ["Original", "Karaoke", "Different"]
)

plt.yticks(
    [0, 1, 2],
    ["Original", "Karaoke", "Different"]
)

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            round(matrix[i, j], 2),
            ha="center",
            va="center",
            fontsize=14
        )

plt.title("Correlation Matrix")
plt.show()

cross = correlate(original, karaoke, mode="full")

plt.figure(figsize=(12, 4))
plt.plot(cross, color="purple")
plt.title("Cross Correlation: Original vs Karaoke")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.show()
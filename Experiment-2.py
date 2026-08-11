import numpy as np
import matplotlib.pyplot as plt

def linear_convolution(x, h):
    return np.convolve(x, h, mode='full')

def circular_convolution(x, h):
    n = max(len(x), len(h))
    X = np.fft.fft(x, n)
    H = np.fft.fft(h, n)
    return np.real(np.fft.ifft(X * H))

x = np.array([3, 1, -2, 4])
h = np.array([2, -1, 3])

linear = linear_convolution(x, h)
circular = circular_convolution(x, h)

plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
markerline, stemlines, baseline = plt.stem(linear)
plt.setp(markerline, color='red', marker='o', markersize=8)
plt.setp(stemlines, color='blue', linewidth=2)
plt.setp(baseline, color='black')
plt.title("Linear Convolution", fontsize=14, color='darkblue')
plt.xlabel("Sample Index", color='purple')
plt.ylabel("Amplitude", color='purple')
plt.grid(color='lightgray', linestyle='--')

plt.subplot(2, 1, 2)
markerline, stemlines, baseline = plt.stem(circular)
plt.setp(markerline, color='green', marker='D', markersize=8)
plt.setp(stemlines, color='orange', linewidth=2)
plt.setp(baseline, color='black')
plt.title("Circular Convolution", fontsize=14, color='darkgreen')
plt.xlabel("Sample Index", color='brown')
plt.ylabel("Amplitude", color='brown')
plt.grid(color='lightblue', linestyle='--')

plt.tight_layout()
plt.show()
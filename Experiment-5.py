import numpy as np
import matplotlib.pyplot as plt

signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

fft_result = np.fft.fft(signal)

magnitude_spectrum = np.abs(fft_result)
phase_spectrum = np.angle(fft_result)

reconstructed_signal = np.fft.ifft(fft_result)

print("Original Signal:", signal)
print("FFT Result:", fft_result)
print("Magnitude Spectrum:", magnitude_spectrum)
print("Phase Spectrum:", phase_spectrum)
print("Reconstructed Signal:", np.real(reconstructed_signal))

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(signal)
plt.title("Original Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.stem(magnitude_spectrum)
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency Index")
plt.ylabel("Magnitude")

plt.subplot(3, 1, 3)
plt.stem(np.real(reconstructed_signal))
plt.title("Reconstructed Signal using IFFT")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
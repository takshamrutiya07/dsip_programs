import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fft import fft, ifft
from scipy.io import wavfile

# ==========================
# Load WAV Audio
# ==========================

sample_rate, samples = wavfile.read("see-you-next-christmas.wav")

# Convert stereo to mono if necessary
if len(samples.shape) == 2:
    samples = samples.mean(axis=1)

samples = samples.astype(np.float32)

# Normalize
samples = samples / np.max(np.abs(samples))

# ==========================
# Impulse Responses
# ==========================

impulse_responses = {
    # Original signal
    "IR1_Original": np.array([1.0], dtype=np.float32),

    # Echo after 100 ms
    "IR2": np.concatenate([
        np.array([1.0]),
        np.zeros(4400),
        np.array([0.7])
    ]).astype(np.float32),

    # Echo after 250 ms
    "IR3": np.concatenate([
        np.array([1.0]),
        np.zeros(11025),
        np.array([0.6])
    ]).astype(np.float32),

    # Multiple echoes
    "IR4": np.concatenate([
        np.array([1.0]),
        np.zeros(4000),
        np.array([0.8]),
        np.zeros(4000),
        np.array([0.6]),
        np.zeros(4000),
        np.array([0.4])
    ]).astype(np.float32),

    # Moving average (low-pass)
    "IR5": np.ones(101, dtype=np.float32) / 101
}
# ==========================
# Process
# ==========================

for name, kernel in impulse_responses.items():

    print(f"Processing {name}")

    # ---------------------
    # Convolution
    # ---------------------

    convoluted = convolve(samples, kernel, mode="same")

    convoluted = convoluted / np.max(np.abs(convoluted))

    # ---------------------
    # Inverse Filter
    # ---------------------

    Y = fft(convoluted)

    H = fft(kernel, len(convoluted))

    epsilon = 1e-6

    X = Y / (H + epsilon)

    recovered = np.real(ifft(X))

    recovered = recovered / np.max(np.abs(recovered))

    # ---------------------
    # Save Audio
    # ---------------------

    wavfile.write(
        f"{name}_Convoluted.wav",
        sample_rate,
        (convoluted * 32767).astype(np.int16)
    )

    wavfile.write(
        f"{name}_Recovered.wav",
        sample_rate,
        (recovered * 32767).astype(np.int16)
    )

    # ---------------------
    # Plot
    # ---------------------

    plt.figure(figsize=(14,8))

    plt.subplot(3,1,1)
    plt.plot(samples)
    plt.title("Original Audio")

    plt.subplot(3,1,2)
    plt.plot(convoluted)
    plt.title(f"Convolution using {name}")

    plt.subplot(3,1,3)
    plt.plot(recovered)
    plt.title("Recovered using Inverse Filtering")

    plt.tight_layout()
    plt.show()

print("Experiment Completed Successfully!")
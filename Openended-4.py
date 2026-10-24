
import os
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import butter, sosfiltfilt

BASE_DIR = Path(__file__).resolve().parent
filename = BASE_DIR / "Original.mp3"
output_dir = BASE_DIR / "output"

output_dir.mkdir(exist_ok=True)

if not filename.exists():
    raise FileNotFoundError(
        f"MP3 file not found: {filename}"
    )

y, sr = librosa.load(
    str(filename),
    sr=None,
    mono=True
)

print("Audio loaded successfully!")
print("Sample Rate:", sr, "Hz")
print("Duration:", round(len(y) / sr, 2), "seconds")


def save_audio(name, audio):
    output_path = output_dir / name

    sf.write(str(output_path), audio, sr)

    print("Saved:", output_path.name)


def lowpass_filter(signal, cutoff, fs, order=6):
    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


# Q3 (a): LOW-PASS FILTER

cutoffs = [500, 1500, 3500]
lowpass_results = {}

print("\nApplying Low-Pass Filters...")

for cutoff in cutoffs:

    if cutoff >= sr / 2:
        raise ValueError(
            f"Cutoff {cutoff} Hz must be below "
            f"Nyquist frequency {sr / 2} Hz"
        )

    filtered_audio = lowpass_filter(
        y, cutoff, sr
    )

    lowpass_results[cutoff] = filtered_audio

    save_audio(
        f"lowpass_{cutoff}Hz.wav",
        filtered_audio
    )


# Q3 (b): BAND-PASS FILTER

def bandpass_filter(
    signal, lowcut, highcut, fs, order=6
):
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


lowcut = 300
highcut = 3400

if lowcut <= 0 or highcut >= sr / 2:
    raise ValueError(
        "Band-pass cutoff frequencies must be "
        "positive and below the Nyquist frequency."
    )

if lowcut >= highcut:
    raise ValueError(
        "Low cutoff must be less than high cutoff."
    )

print("\nApplying Band-Pass Filter...")

# Estimate vocal-frequency components
vocal_audio = bandpass_filter(
    y, lowcut, highcut, sr
)

# Estimate remaining frequency components
background_audio = y - vocal_audio

save_audio(
    "estimated_vocals.wav",
    vocal_audio
)

save_audio(
    "estimated_background.wav",
    background_audio
)


# SPECTROGRAM FUNCTION

def plot_spectrogram(signal, title, ax):

    stft = librosa.stft(signal)

    db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    librosa.display.specshow(
        db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        cmap="viridis",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")


# LOW-PASS SPECTROGRAMS

print("\nGenerating Low-Pass Spectrograms...")

fig, axes = plt.subplots(
    4, 1, figsize=(12, 14)
)

plot_spectrogram(
    y,
    "Original Audio",
    axes[0]
)

for i, cutoff in enumerate(cutoffs):

    plot_spectrogram(
        lowpass_results[cutoff],
        f"Low-Pass Filter: {cutoff} Hz",
        axes[i + 1]
    )

plt.tight_layout()

plt.savefig(
    output_dir / "lowpass_spectrogram.png",
    dpi=150
)

plt.close(fig)


# BAND-PASS SPECTROGRAMS

print("\nGenerating Band-Pass Spectrograms...")

fig, axes = plt.subplots(
    3, 1, figsize=(12, 11)
)

plot_spectrogram(
    y,
    "Original Audio",
    axes[0]
)

plot_spectrogram(
    vocal_audio,
    "Estimated Vocal Frequencies (300-3400 Hz)",
    axes[1]
)

plot_spectrogram(
    background_audio,
    "Estimated Background Frequencies",
    axes[2]
)

plt.tight_layout()

plt.savefig(
    output_dir / "bandpass_spectrogram.png",
    dpi=150
)

plt.close(fig)


# DISPLAY OUTPUT FILES

print("\nQ3 (a): Low-Pass Filter Outputs")
print("1. lowpass_500Hz.wav")
print("2. lowpass_1500Hz.wav")
print("3. lowpass_3500Hz.wav")

print("\nQ3 (b): Band-Pass Filter Outputs")
print("4. estimated_vocals.wav")
print("5. estimated_background.wav")

print("\nSpectrogram Images:")
print("6. lowpass_spectrogram.png")
print("7. bandpass_spectrogram.png")

print("\nAll files saved in:")
print(output_dir)

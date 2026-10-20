import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

Fs = 10000          
N = 31              
nyquist = Fs / 2  

fc_low = 3000
fc_high = 6000

print("=" * 60)
print("FIR FILTER DESIGN EXPERIMENT")
print("=" * 60)
print(f"Sampling Frequency : {Fs / 1000:.1f} kHz")
print(f"Number of Taps     : {N}")
print(f"Nyquist Frequency  : {nyquist / 1000:.1f} kHz")
print("Window             : Hamming")
print()

if fc_high >= nyquist:
    print("WARNING: 6 kHz is not a valid digital cutoff!")
    print("Nyquist frequency is 5 kHz.")
    print("Using 4 kHz as a temporary valid upper cutoff")
    print("for the band-pass and band-stop demonstrations.")
    print("Please confirm this correction with your teacher.")
    print()
    fc_high_used = 4000
else:
    fc_high_used = fc_high
low_pass = firwin(
    N,
    cutoff=fc_low,
    fs=Fs,
    window="hamming",
    pass_zero="lowpass"
)
high_pass = firwin(
    N,
    cutoff=fc_low,
    fs=Fs,
    window="hamming",
    pass_zero="highpass"
)
band_pass = firwin(
    N,
    cutoff=[fc_low, fc_high_used],
    fs=Fs,
    window="hamming",
    pass_zero="bandpass"
)
band_stop = firwin(
    N,
    cutoff=[fc_low, fc_high_used],
    fs=Fs,
    window="hamming",
    pass_zero="bandstop"
)
filters = [
    ("Low-Pass Filter", low_pass),
    ("High-Pass Filter", high_pass),
    ("Band-Pass Filter", band_pass),
    ("Band-Stop Filter", band_stop)
]

for name, coefficients in filters:
    print("-" * 60)
    print(name)
    print("-" * 60)
    print("Filter coefficients:")
    print(np.array2string(
        coefficients,
        precision=5,
        suppress_small=True,
        max_line_width=100
    ))
    print()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "FIR Filter Frequency Responses (Hamming Window)",
    fontsize=15
)

for ax, (name, coefficients) in zip(axes.flat, filters):
    frequencies, response = freqz(
        coefficients,
        worN=2048,
        fs=Fs
    )

    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-8)
    )

    ax.plot(frequencies / 1000, magnitude_db)
    ax.set_title(name)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, Fs / 2000)
    ax.set_ylim(-80, 5)
    ax.grid(True)

plt.tight_layout()
plt.show()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("1. The FIR low-pass filter passes low frequencies.")
print("2. The FIR high-pass filter passes high frequencies.")
print("3. The FIR band-pass filter passes a selected frequency band.")
print("4. The FIR band-stop filter rejects a selected frequency band.")
print("All filters use 31 coefficients and a Hamming window.")
print("The band-pass and band-stop upper cutoff was adjusted")
print("to 4 kHz because 6 kHz exceeds the 5 kHz Nyquist limit.")
print("Confirm the intended cutoff with your teacher.")

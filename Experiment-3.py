import numpy as np
import matplotlib.pyplot as plt

def cross_correlation(signal1, signal2):
    return np.correlate(signal1, signal2, mode='full')

def autocorrelation(signal):
    return np.correlate(signal, signal, mode='full')

# Input Signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])

# Calculate Correlations
cross_corr = cross_correlation(signal1, signal2)
auto_corr = autocorrelation(signal1)

# Print Results
print("Signal 1:")
print(signal1)

print("\nSignal 2:")
print(signal2)

print("\nCross Correlation:")
print(cross_corr)

print("\nAutocorrelation:")
print(auto_corr)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 7), facecolor="#F5F5F5")

# Cross Correlation Plot
plt.subplot(2, 1, 1)
markerline, stemlines, baseline = plt.stem(cross_corr)
plt.setp(markerline, marker='o', markersize=8, color='royalblue')
plt.setp(stemlines, color='royalblue', linewidth=2)
plt.setp(baseline, color='black')

plt.title("Cross Correlation", fontsize=16, fontweight='bold')
plt.xlabel("Time Lag", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(["Cross Correlation"], loc='upper left')

# Autocorrelation Plot
plt.subplot(2, 1, 2)
markerline, stemlines, baseline = plt.stem(auto_corr)
plt.setp(markerline, marker='s', markersize=8, color='crimson')
plt.setp(stemlines, color='crimson', linewidth=2)
plt.setp(baseline, color='black')

plt.title("Autocorrelation", fontsize=16, fontweight='bold')
plt.xlabel("Time Lag", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(["Autocorrelation"], loc='upper left')

plt.tight_layout()
plt.show()
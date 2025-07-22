import os
import csv
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.signal as ssig

# Settings
AUDIO_PATH    = "eurasian_blackbird.wav"
OUTPUT_DIR    = "output_segments"
CSV_PATH      = os.path.join(OUTPUT_DIR, "segments.csv")
SR            = 22050        # sample rate
MIN_DURATION  = 0.05         # minimum segment length (s)
MIN_INTERVAL  = 0.1          # minimum time between onsets (s)
ENERGY_PCTL   = 30           # drop lowest 30% by energy
PREFIX        = "note"

# Load & pre‐filter
y, sr = librosa.load(AUDIO_PATH, sr=SR)
b, a = ss.butter(4, 500, btype="highpass", fs=sr)
y = ss.filtfilt(b, a, y)

# Onset‐strength (spectral flux)
hop_length = 256
n_fft      = 1024
onset_env  = librosa.onset.onset_strength(
    y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, aggregate=np.median
)

# Peak‐pick the big flux jumps
# dynamic threshold = mean + 1*std
thr = onset_env.mean() + onset_env.std()
# enforce at least MIN_INTERVAL seconds between peaks
min_dist = int((MIN_INTERVAL * sr) / hop_length)

peaks, props = ssig.find_peaks(
    onset_env,
    height=thr,
    distance=min_dist
)
onset_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
print(f"Picked {len(onset_times)} strong onsets")

# Plot for sanity check
times_env = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
for t in onset_times:
    plt.axvline(t, color="r", linestyle="--", alpha=0.8)
plt.title("Waveform with Strong Onsets")

plt.subplot(2,1,2)
plt.plot(times_env, onset_env, label="Spectral Flux")
plt.hlines(thr, times_env[0], times_env[-1],
           color="gray", linestyle="--", label="Threshold")
plt.vlines(onset_times, 0, onset_env.max(),
           color="r", linestyle="--")
plt.legend(loc="upper right")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# Segment & energy filter
segments, seg_times, seg_energy = [], [], []
for i in range(len(onset_times)-1):
    s, e = onset_times[i], onset_times[i+1]
    if e-s < MIN_DURATION: continue
    s_samp, e_samp = int(s*sr), int(e*sr)
    seg = y[s_samp:e_samp]
    segments.append(seg)
    seg_times.append((s,e))
    seg_energy.append(np.sum(seg**2))

# tail
if onset_times.size and (len(y)/sr - onset_times[-1] >= MIN_DURATION):
    s = onset_times[-1]
    seg = y[int(s*sr):]
    segments.append(seg)
    seg_times.append((s, len(y)/sr))
    seg_energy.append(np.sum(seg))

print(f"Found {len(segments)} raw segments")

# drop lowest-energy
pctl = np.percentile(seg_energy, ENERGY_PCTL)
filtered_segs, filtered_times = [], []
for seg, (s,e), en in zip(segments, seg_times, seg_energy):
    if en >= pctl:
        filtered_segs.append(seg)
        filtered_times.append((s,e))
print(f"Kept {len(filtered_segs)} segments (top {100-ENERGY_PCTL}%)")

# Export
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename","start","end","duration"])
    for i, (seg, (s,e)) in enumerate(zip(filtered_segs, filtered_times)):
        fn = f"{PREFIX}_{i:03d}.wav"
        sf.write(os.path.join(OUTPUT_DIR, fn), seg, sr)
        w.writerow([fn, round(s,3), round(e,3), round(e-s,3)])
        print("Exported", fn)

print("Done.")
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict

# Configuration
SEG_DIR    = "output_segments"
CSV_PATH   = os.path.join(SEG_DIR, "segments.csv")
GIF_PATH   = "notes_clustered.gif"
FPS        = 24
TIME_SCALE = 1    # 1 = real time

# Load metadata
df        = pd.read_csv(CSV_PATH)
filenames = df["filename"].values
starts    = df["start"].values
ends      = df["end"].values
durations = df["duration"].values

# Feature names
feature_names = (
    ["onset","duration","energy","rms","zcr",
     "centroid","bandwidth","rolloff"] +
    [f"contrast_{i}" for i in range(7)] +
    [f"mfcc_{i}"     for i in range(13)] +
    [f"chroma_{i}"   for i in range(12)] +
    [f"tonnetz_{i}"  for i in range(6)]
)

# Feature extraction
def extract_features(seg, sr, onset, end):
    duration = end - onset
    energy   = np.sum(seg**2)
    rms      = np.mean(librosa.feature.rms(y=seg)[0])
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=seg)[0])
    centroid = np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)[0])
    bandwidth= np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)[0])
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr)[0])
    contrast = np.mean(librosa.feature.spectral_contrast(y=seg, sr=sr), axis=1)
    mfccs    = np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13), axis=1)
    chroma   = np.mean(librosa.feature.chroma_stft(y=seg, sr=sr), axis=1)
    tonnetz  = np.mean(librosa.feature.tonnetz(y=seg, sr=sr), axis=1)
    return np.hstack([
        onset, duration, energy, rms, zcr,
        centroid, bandwidth, rolloff,
        contrast, mfccs, chroma, tonnetz
    ])

# Build feature matrix
features = []
for fn, onset, end in zip(filenames, starts, ends):
    seg, sr = librosa.load(os.path.join(SEG_DIR, fn), sr=None)
    features.append(extract_features(seg, sr, onset, end))
X = np.vstack(features)

# Preprocess & PCA
X_scaled = StandardScaler().fit_transform(X)
sel      = VarianceThreshold(0.5)
X_sel    = sel.fit_transform(X_scaled)

selected_names = [n for n, keep in zip(feature_names, sel.get_support()) if keep]
pca = PCA(n_components=3)
X3  = pca.fit_transform(X_sel)

def top_feature(comp, names):
    return names[np.argmax(np.abs(comp))]

label_x = top_feature(pca.components_[0], selected_names)
label_y = top_feature(pca.components_[1], selected_names)
label_z = top_feature(pca.components_[2], selected_names)

# Silhouette analysis (k from 3 to 12)
ks, sil_scores = range(3,15), []
for k in ks:
    km  = KMeans(n_clusters=k, random_state=42).fit(X_sel)
    sil = silhouette_score(X_sel, km.labels_)
    sil_scores.append(sil)
    print(f"k={k:2d}  silhouette={sil:.4f}")

plt.figure(figsize=(6,3))
plt.plot(ks, sil_scores, '-o', color='steelblue')
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()

best_k = ks[np.argmax(sil_scores)]
print(f"\nBest k by silhouette: {best_k}\n")

# Cluster with best_k
n_clusters  = best_k
kmeans      = KMeans(n_clusters=n_clusters, random_state=42)
note_labels = kmeans.fit_predict(X_sel)

# Map times to frames
scaled_time = (starts - starts.min()) * TIME_SCALE
frame_idx   = (scaled_time * FPS).astype(int)
max_frame   = frame_idx.max() + int(1.5 * FPS)

frame_notes = [[] for _ in range(max_frame)]
for i, f0 in enumerate(frame_idx):
    for f in range(f0, max_frame):
        frame_notes[f].append(i)

# Detect top 5 motifs & their completion frames
sorted_idx   = np.argsort(starts)
symbolic_seq = note_labels[sorted_idx]
motif_len    = 3
all_motifs   = [tuple(symbolic_seq[i:i+motif_len])
                for i in range(len(symbolic_seq)-motif_len+1)]
counts       = Counter(all_motifs)
top5         = counts.most_common(5)

# For each occurrence, record the frame when the motif completes
events = defaultdict(list)  # frame -> list of motif indices
for m_i, (motif, _) in enumerate(top5):
    for i in range(len(symbolic_seq)-motif_len+1):
        if tuple(symbolic_seq[i:i+motif_len]) == motif:
            idx_seq      = sorted_idx[i:i+motif_len]
            finish_frame = frame_idx[idx_seq].max()
            events[finish_frame].append(m_i)

# Prepare figure & dynamic motif board
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')

# Initial counts = 0, prepare text artists
display_counts = [0]*len(top5)
motif_texts    = []
for i,(motif, _) in enumerate(top5):
    line = f"{i+1}. {' → '.join(map(str,motif))} (0)"
    txt  = ax.text2D(-0.1, 0.95 - i*0.05, line,
                     transform=ax.transAxes,
                     fontsize=8, color='black', va='top')
    motif_texts.append(txt)

# Track last highlighted motif for flashing
last_activated = None

# 3D axes setup
ax.set_xlim(X3[:,0].min()*1.1, X3[:,0].max()*1.1)
ax.set_ylim(X3[:,1].min()*1.1, X3[:,1].max()*1.1)
ax.set_zlim(X3[:,2].min()*1.1, X3[:,2].max()*1.1)
ax.set_xlabel(label_x)
ax.set_ylabel(label_y)
ax.set_zlabel(label_z)
ax.set_title("Singing of Eurasian Blackbird")

# discrete colormap matching cluster count
if n_clusters <= 10:
    cmap = colormaps["tab10"].resampled(n_clusters)
else:
    cmap = colormaps["tab20"].resampled(n_clusters)

# dummy scatter for colorbar
sc0 = ax.scatter([], [], [], c=[], cmap=cmap,
                 vmin=0, vmax=n_clusters-1)
cbar = fig.colorbar(sc0, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label("Cluster ID")

texts = []

def update(frame):
    global last_activated
    ids = frame_notes[frame]
    # clear old points & texts
    for coll in list(ax.collections): coll.remove()
    for txt in texts: txt.remove()
    texts.clear()

    # handle motif completion events
    if frame in events:
        for m_i in events[frame]:
            display_counts[m_i] += 1
            motif, _ = top5[m_i]
            new_line = f"{m_i+1}. {' → '.join(map(str,motif))} ({display_counts[m_i]})"
            motif_texts[m_i].set_text(new_line)

        # flash current motif in red, reset previous
        current = events[frame][-1]
        if last_activated is not None and last_activated != current:
            motif_texts[last_activated].set_color('black')
        motif_texts[current].set_color('red')
        last_activated = current

    if not ids:
        return []

    xi = X3[ids,0]; yi = X3[ids,1]; zi = X3[ids,2]
    ci = note_labels[ids]
    si = 20 + 120 * durations[ids] / durations.max()

    sc = ax.scatter(xi, yi, zi,
                    c=ci, cmap=cmap,
                    vmin=0, vmax=n_clusters-1,
                    s=si, alpha=0.9)

    for j, idx in enumerate(ids):
        t = ax.text(xi[j], yi[j], zi[j],
                    str(idx+1), fontsize=7, alpha=0.6)
        texts.append(t)

    # return scatter, note texts, and motif board texts
    return [sc] + texts + motif_texts

ani = FuncAnimation(fig, update,
                    frames=max_frame,
                    interval=1000/FPS)
ani.save(GIF_PATH, writer="pillow", fps=FPS)
print(f"Animation saved: {GIF_PATH}")
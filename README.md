#🎨PixelReduce — Image Compression via K-Means Color Quantization

Reduce an image to a handful of colors using K-Means clustering on pixel RGB
values — a small, visual demonstration of unsupervised learning applied to
lossy image compression.

Every pixel in an image is a point in 3D RGB space. PixelReduce clusters all
of an image's pixels into **K** color groups, then replaces each pixel with
its cluster's centroid color, producing a compressed-palette version of the
original. Fewer colors → smaller palette → smaller file, at a measurable cost
in visual quality.

---

## ✨ Features

- Upload any image and choose **K**, the number of colors (2–64)
- Before/after visual comparison
- **PSNR** (Peak Signal-to-Noise Ratio) as a quality metric
- Estimated file-size comparison (original vs. palette-indexed encoding)
- Extracted color palette with hex values
- **Posterize mode** — a stylized, higher-contrast variant of the palette
- Optional side-by-side comparison against PIL's built-in **median-cut**
  quantizer

---

## 🧠 How it works

1. **Points in RGB space** — every pixel's `(R, G, B)` triple becomes a
   coordinate in a 3D color cube. A modest image can easily contain tens of
   thousands of such points.
2. **Assign & update** — K centroids are seeded with K-Means++. Each pixel
   is assigned to its nearest centroid; each centroid then moves to the mean
   of its assigned pixels. This repeats until the centroids stop moving
   (or a max iteration count is hit).
3. **Repaint with centroids** — once converged, every pixel is replaced by
   its cluster's centroid color, so the final image uses only K colors
   instead of potentially millions.

---

## 📦 Project structure

```
PixelReduce/
├── app.py              # Streamlit + scikit-learn backend (the real implementation)
├── requirements.txt     # Python dependencies
├── SETUP.md              # Step-by-step VS Code setup guide
├── pixelreduce.html      # Standalone browser demo (vanilla JS, zero install)
└── README.md
```

- **`app.py`** is the reference implementation: it runs true `sklearn.cluster.KMeans`
  on pixel data, computes PSNR and size-reduction metrics, and renders
  everything through a Streamlit UI.
- **`pixelreduce.html`** is a self-contained, dependency-free browser demo
  (K-Means implemented in vanilla JS) — useful for quickly showing the concept
  without installing anything.

---

## 🚀 Getting started

### Requirements
- Python 3.9+

### Install & run

```bash
git clone https://github.com/<your-username>/PixelReduce.git
cd PixelReduce

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

Full step-by-step instructions, including VS Code-specific setup, are in
[`SETUP.md`](./SETUP.md).

---

## 🛠️ Tech stack

| Purpose | Library |
|---|---|
| Clustering | `scikit-learn` (`KMeans`, K-Means++ initialization) |
| Image I/O | `Pillow` |
| Numerical ops | `NumPy` |
| Web UI | `Streamlit` |

---

## 📊 Example metrics

| K | PSNR (dB) | Notes |
|---|---|---|
| 4  | ~18–22 | Strong posterization, very small palette |
| 8  | ~24–28 | Visible banding but recognizable |
| 16 | ~28–32 | Good balance of size vs. quality |
| 32+ | ~32–36+ | Close to visually lossless for most photos |

*(Actual values depend heavily on the source image.)*

---

## 🗺️ Roadmap

- [ ] Benchmark against median-cut and octree quantization more rigorously
- [ ] Extend to GIF and video frame-sequence compression
- [ ] Combine with clustering-based background removal into a fuller AI
      image toolkit

---

## 🎓 Background

This project was built as an exploration of applying an unsupervised
learning algorithm (K-Means) to a tangible, visual problem — image
compression — comparing cluster count, reconstruction quality, and
estimated output size.

Team Members
This project was developed as a group project by:
Sampada K.C.
Shine Pandey
Smarita Karkee
Tabita Mali


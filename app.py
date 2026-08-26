"""
PixelReduce — Image Compression via K-Means Color Quantization
=================================================================

A Streamlit app that clusters an image's pixels in RGB space with
scikit-learn's KMeans, replaces every pixel with its cluster's centroid
color, and reports the resulting size and quality trade-off.

Run it with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import time

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="PixelReduce — K-Means Color Quantization",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 PixelReduce")
st.caption(
    "Every pixel is a point in 3D RGB space. K-Means clusters all pixels "
    "into **K** color groups, then each pixel is replaced by its cluster's "
    "centroid color — a compressed-palette version of the original image."
)


# --------------------------------------------------------------------------
# Core algorithm
# --------------------------------------------------------------------------

def quantize_image(image: Image.Image, k: int, sample_size: int = 20_000,
                    random_state: int = 42):
    """
    Cluster an image's pixels into k colors with K-Means and repaint the
    image using the cluster centroids.

    To keep large images fast, KMeans is *fit* on a random sample of
    pixels (this is standard practice — cluster centers stabilize well
    before you've seen every pixel), then every pixel in the full-resolution
    image is assigned to its nearest fitted centroid.

    Returns
    -------
    quantized : np.ndarray (H, W, 3) uint8
        The recolored image.
    palette : np.ndarray (k, 3) uint8
        The K centroid colors.
    n_iter : int
        Number of iterations KMeans took to converge.
    """
    arr = np.array(image.convert("RGB"))
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3).astype(np.float64)

    # Fit on a random subsample for speed on large images.
    n_pixels = pixels.shape[0]
    if n_pixels > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n_pixels, size=sample_size, replace=False)
        fit_data = pixels[sample_idx]
    else:
        fit_data = pixels

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=4,
        random_state=random_state,
    )
    kmeans.fit(fit_data)

    # Assign every pixel (full resolution) to its nearest centroid.
    labels = kmeans.predict(pixels)
    palette = kmeans.cluster_centers_.round().astype(np.uint8)

    quantized = palette[labels].reshape(h, w, 3)
    return quantized, palette, kmeans.n_iter_


def posterize_palette(palette: np.ndarray, saturation: float = 1.35,
                       contrast: float = 1.35) -> np.ndarray:
    """Push the extracted palette toward a bolder, poster-like look."""
    palette = palette.astype(np.float64)
    avg = palette.mean(axis=1, keepdims=True)
    saturated = avg + (palette - avg) * saturation
    contrasted = 128 + (saturated - 128) * contrast
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def apply_palette(image: Image.Image, palette: np.ndarray) -> np.ndarray:
    """Recolor an image by nearest-centroid lookup against a given palette
    (used to re-render with the posterized palette without re-clustering)."""
    arr = np.array(image.convert("RGB")).astype(np.float64)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    # (n_pixels, k) distance matrix — fine at these image sizes.
    dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    return palette[labels].reshape(h, w, 3)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two same-shape uint8 images."""
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def estimate_indexed_size(width: int, height: int, k: int) -> int:
    """
    Estimate the size of a palette-indexed encoding (GIF/PNG-8 style):
    a K x 3 palette table plus `bits_per_pixel = ceil(log2(k))` per pixel,
    plus a small fixed header.
    """
    bits_per_pixel = max(1, int(np.ceil(np.log2(max(k, 2)))))
    pixel_bytes = (width * height * bits_per_pixel) / 8
    palette_bytes = k * 3
    header_bytes = 54
    return int(pixel_bytes + palette_bytes + header_bytes)


def median_cut_baseline(image: Image.Image, k: int) -> np.ndarray:
    """
    PIL's built-in median-cut quantizer, used as a quick comparison point
    against K-Means (see "Possible Improvements" in the project brief).
    """
    quant = image.convert("RGB").quantize(colors=k, method=Image.MEDIANCUT)
    return np.array(quant.convert("RGB"))


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    k = st.slider("K — number of colors", min_value=2, max_value=64, value=8)
    max_dim = st.slider("Max working dimension (px)", 128, 1024, 480, step=32,
                         help="Larger images are downscaled before clustering to keep this responsive.")
    posterize = st.checkbox("Posterize mode (artistic)", value=False)
    show_median_cut = st.checkbox("Compare against median-cut quantization", value=False)
    st.caption("Runs scikit-learn `KMeans` on pixel RGB values.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if uploaded is None:
    st.info("Upload an image in the sidebar to get started.")
    st.stop()

original_bytes = uploaded.getvalue()
image = Image.open(io.BytesIO(original_bytes))

# Downscale for responsiveness; note this in the UI so results are honest.
w, h = image.size
scale = min(1.0, max_dim / max(w, h))
work_w, work_h = max(1, int(w * scale)), max(1, int(h * scale))
working_image = image.resize((work_w, work_h)) if scale < 1.0 else image

t0 = time.time()
quantized, palette, n_iter = quantize_image(working_image, k)
elapsed = time.time() - t0

if posterize:
    poster_palette = posterize_palette(palette)
    quantized = apply_palette(working_image, poster_palette)
    palette = poster_palette

original_arr = np.array(working_image.convert("RGB"))

# --- Metrics ---
quality = psnr(original_arr, quantized)
unique_colors = len(np.unique(original_arr.reshape(-1, 3), axis=0))
comp_bytes = estimate_indexed_size(work_w, work_h, k)
orig_bytes = len(original_bytes)
reduction_pct = max(0.0, (1 - comp_bytes / orig_bytes) * 100)

# --- Layout ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(working_image, use_container_width=True)
with col2:
    st.subheader(f"Compressed — K = {k}")
    st.image(quantized, use_container_width=True)

st.divider()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Original size", f"{orig_bytes / 1024:.1f} KB")
m2.metric("Compressed size (est.)", f"{comp_bytes / 1024:.1f} KB")
m3.metric("Size reduction", f"{reduction_pct:.0f}%")
m4.metric("PSNR (quality)", "∞" if np.isinf(quality) else f"{quality:.1f} dB")
m5.metric("KMeans iterations", n_iter)

st.caption(f"Clustering ran in {elapsed:.2f}s at {work_w}×{work_h}px "
           f"({unique_colors:,} unique colors in the original).")

st.subheader("Extracted palette")
swatch_cols = st.columns(k)
for i, color in enumerate(palette):
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    with swatch_cols[i % k]:
        st.markdown(
            f"<div style='background:{hex_color};height:44px;border-radius:6px;"
            f"border:1px solid #333;'></div>"
            f"<div style='font-family:monospace;font-size:11px;text-align:center;'>{hex_color}</div>",
            unsafe_allow_html=True,
        )

# --- Download ---
out_buf = io.BytesIO()
Image.fromarray(quantized).save(out_buf, format="PNG")
st.download_button(
    "Download compressed PNG",
    data=out_buf.getvalue(),
    file_name=f"pixelreduce_k{k}.png",
    mime="image/png",
)

# --- Optional median-cut comparison ---
if show_median_cut:
    st.divider()
    st.subheader("K-Means vs. median-cut")
    mc_image = median_cut_baseline(working_image, k)
    mc_psnr = psnr(original_arr, mc_image)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**K-Means** — PSNR {quality:.1f} dB" if not np.isinf(quality) else "**K-Means** — PSNR ∞")
        st.image(quantized, use_container_width=True)
    with c2:
        st.markdown(f"**Median-cut (PIL)** — PSNR {mc_psnr:.1f} dB" if not np.isinf(mc_psnr) else "**Median-cut** — PSNR ∞")
        st.image(mc_image, use_container_width=True)

    st.caption(
        "K-Means minimizes within-cluster variance directly in RGB space, so it "
        "often preserves dominant colors more faithfully; median-cut is faster "
        "(no iteration) and tends to spread the palette more evenly across the "
        "color space — useful context for the 'Possible Improvements' comparison."
    )

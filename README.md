🎨 # PixelReduce

Image compression via K-Means color quantization — a Streamlit app that clusters an image's pixels in RGB space and repaints each pixel with its cluster's centroid color.

## Features
- Adjustable color count (K = 2–64)
- Auto-downscaling + pixel sampling for responsiveness on large images
- Quality metric (PSNR) and estimated size reduction
- Extracted color palette with hex codes
- Posterize mode for a bolder, artistic look
- Optional comparison against PIL's median-cut quantizer
- One-click PNG download

## How it works
- Downscale image to the working dimension
- Sample pixels (if > 20,000) and fit KMeans(n_clusters=k, init="k-means++")
- Predict cluster labels for every pixel in the full-resolution image
- Repaint each pixel with its cluster centroid color
- Compute PSNR, estimated size, and display the palette

## Tech stack
Streamlit · NumPy · Pillow · scikit-learn

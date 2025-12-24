# In terminal run ./run.sh for linux 

# Image Super-Resolution using Edge-Preserving K-SVD
## Project Overview

**This project implements a patch-based image super-resolution (SR) method using edge-preserving K-SVD dictionaries. It learns separate dictionaries for low-resolution (LR) and high-frequency (HF) image patches, allowing the reconstruction of high-resolution images from low-resolution inputs.**

### The workflow includes:                                                                                                                                                                                                                                                                                

- Reading high-resolution (HR) images.
- Downsampling HR images to generate LR images.
- Computing high-frequency components (HR - blurred HR).
- Extracting patch pairs (LR patch ↔ HF patch) from edge-rich areas.
- Training K-SVD dictionaries:
- D_L for LR patches
- D_H for HF patches using the same sparse coefficients
- Using the learned dictionaries for super-resolution inference.

### Features

- Train K-SVD dictionaries for multiple scales (x2, x3, x4)
- Extract edge-rich patches from HR images
- Generate LR images via bicubic downsampling
- Compute high-frequency (HF) components for reconstruction
- Use sparse coding to reconstruct HR images from LR patches
- Optional backprojection for refinement


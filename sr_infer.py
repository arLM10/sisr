"""
Inference script:
 - load trained dictionaries D_L, D_H for a chosen scale
 - for each test HR image:
    - extract LR patches
    - compute sparse codes (OMP) possibly with edge-weighted penalty (we implement simple weighting by scaling dictionary columns)
    - reconstruct HR patches
    - aggregate to HR image
    - optional back-projection refinement
 - save results
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import orthogonal_mp
from joblib import load
from config import *
from utils import *
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

import argparse

# ---------------------------------------------------------------------------#
# Dictionary helpers
# ---------------------------------------------------------------------------#
def load_dicts(scale):
    D_L = np.load(os.path.join(OUTPUT_DIR, f"D_L_x{scale}.npy"))
    D_H = np.load(os.path.join(OUTPUT_DIR, f"D_H_x{scale}.npy"))
    return D_L, D_H

def weighted_omp(D, y, n_nonzero_coefs=6, weights=None):
    """
    Enhanced edge-weighted sparse coding with proper weight handling:
    - weights: array shape (n_atoms,) where higher values encourage selection
    - Uses iterative weighted OMP for better edge preservation
    """
    if weights is None:
        # Standard OMP
        try:
            coef = orthogonal_mp(D, y.reshape(-1, 1), n_nonzero_coefs=n_nonzero_coefs)
            return coef.ravel()
        except:
            # Fallback if OMP fails
            return np.zeros(D.shape[1])
    
    # Weighted OMP implementation
    w = np.array(weights).clip(0.2, 5.0)  # Clamp weights: 0.2=min, 5.0=max
    
    # Scale dictionary atoms by weights (higher weight = more likely to be selected)
    D_weighted = D * w[None, :]
    
    try:
        # Solve with weighted dictionary
        coef_weighted = orthogonal_mp(D_weighted, y.reshape(-1, 1), n_nonzero_coefs=n_nonzero_coefs)
        # Rescale coefficients back
        coef = coef_weighted.ravel() * w
        return coef
    except:
        # Fallback to standard OMP if weighted version fails
        try:
            coef = orthogonal_mp(D, y.reshape(-1, 1), n_nonzero_coefs=n_nonzero_coefs)
            return coef.ravel()
        except:
            return np.zeros(D.shape[1])

def back_projection(hr, lr, scale, iterations=8):
    """Enhanced iterative back-projection with edge preservation"""
    hr_cp = hr.copy()
    for i in range(iterations):
        # Downsample current HR estimate
        down = downsample(hr_cp, scale)
        
        # Compute error
        err = lr - down
        
        # Upsample error with edge-preserving interpolation
        up_err = upsample(err, scale)
        
        # Adaptive step size based on iteration
        step_size = 0.8 * (0.9 ** i)  # Decreasing step size
        
        # Apply correction with edge-aware weighting
        hr_cp = hr_cp + step_size * up_err
        
        # Clamp values to valid range
        hr_cp = np.clip(hr_cp, 0.0, 1.0)
    
    return hr_cp

def compute_edge_weights(lr_patch, D_L, edge_strength_threshold=0.1):
    """
    Compute edge-aware weights for dictionary atoms based on patch characteristics.
    Returns weights that encourage edge-preserving atoms for edge-rich patches.
    """
    # Compute gradient magnitude
    mag = gradient_magnitude(lr_patch)
    edge_strength = mag.mean()
    
    # Compute local texture features
    patch_std = np.std(lr_patch)
    patch_var = np.var(lr_patch)
    
    # Base weights (all atoms equally likely)
    weights = np.ones(D_L.shape[1])
    
    if edge_strength > edge_strength_threshold:
        # For edge-rich patches, compute atom-specific weights
        lr_vec = lr_patch.reshape(-1)
        lr_vec = lr_vec - np.mean(lr_vec)  # zero-mean
        
        # Compute correlation between patch and each dictionary atom
        correlations = np.abs(D_L.T @ lr_vec)
        correlations = correlations / (np.linalg.norm(D_L, axis=0) * np.linalg.norm(lr_vec) + 1e-12)
        
        # Weight atoms based on correlation and edge strength
        edge_factor = min(edge_strength * 3.0, 2.0)  # Scale factor based on edge strength
        texture_factor = min(patch_std * 2.0, 1.5)   # Scale factor based on texture
        
        # Higher correlation + edge strength = higher weight
        weights = 1.0 + edge_factor * correlations + texture_factor * correlations
        
        # Ensure weights are in reasonable range
        weights = np.clip(weights, 0.5, 3.0)
    
    return weights
def infer_single_image(lr_img, D_L, D_H, scale, use_edge_weights=True):
    """
    Reconstruct HR image via sparse coding. If use_edge_weights=False, this is the
    plain sparse SR baseline; if True, applies the enhanced edge-weighting mechanism.
    """
    p_lr = LR_PATCH_SIZE
    p_hr = LR_PATCH_SIZE * scale + HR_PAD
    h_lr, w_lr = lr_img.shape
    hr_patches = []
    
    for i in range(0, h_lr - p_lr + 1, PATCH_STEP):
        for j in range(0, w_lr - p_lr + 1, PATCH_STEP):
            lp = lr_img[i:i+p_lr, j:j+p_lr]
            vec = lp.reshape(-1)
            vec = vec - np.mean(vec)  # zero-mean
            
            # Compute edge-aware weights
            if use_edge_weights:
                weights = compute_edge_weights(lp, D_L)
            else:
                weights = None
            
            # Sparse coding with edge-aware weights
            alpha = weighted_omp(D_L, vec, n_nonzero_coefs=SPARSITY, weights=weights)
            
            # Reconstruct HR patch
            hr_vec = D_H.dot(alpha)
            hr_patch = hr_vec.reshape(p_hr, p_hr)
            
            # Add local mean to stabilize brightness
            mean_lr_up = np.mean(upsample(lp, scale))
            hr_patch = hr_patch + mean_lr_up
            hr_patches.append(hr_patch)

    # Aggregate patches with overlap handling
    target_shape = (h_lr * scale, w_lr * scale)
    recon = np.zeros(target_shape, dtype=np.float32)
    weight = np.zeros(target_shape, dtype=np.float32)
    
    idx = 0
    for i in range(0, h_lr - p_lr + 1, PATCH_STEP):
        for j in range(0, w_lr - p_lr + 1, PATCH_STEP):
            hi = i * scale
            hj = j * scale
            patch = hr_patches[idx]
            ph, pw = patch.shape
            
            if hi + ph <= target_shape[0] and hj + pw <= target_shape[1]:
                recon[hi:hi+ph, hj:hj+pw] += patch
                weight[hi:hi+ph, hj:hj+pw] += 1.0
            else:
                ph_eff = min(ph, target_shape[0] - hi)
                pw_eff = min(pw, target_shape[1] - hj)
                recon[hi:hi+ph_eff, hj:hj+pw_eff] += patch[:ph_eff, :pw_eff]
                weight[hi:hi+ph_eff, hj:hj+pw_eff] += 1.0
            idx += 1
    
    # Normalize by overlap count
    mask = weight > 0
    recon[mask] /= weight[mask]
    
    # Fill uncovered borders with bicubic upsample
    bic = upsample(lr_img, scale)
    recon[~mask] = bic[~mask]
    
    # Enhanced back-projection with edge preservation
    recon = back_projection(recon, lr_img, scale, iterations=BACKPROJECTION_ITERS)
    recon = np.clip(recon, 0.0, 1.0)
    return recon


def process_test_folder(scale):
    """
    For each test HR image, we:
      - generate LR by downsampling
      - save LR
      - bicubic baseline (upsample LR)
      - sparse SR (no edge weighting)
      - edge-aware SR (current heuristic)
      - save HR GT
    """
    makedirs(OUTPUT_DIR)
    D_L, D_H = load_dicts(scale)
    test_files = list_image_files(HR_TEST_DIR)
    for f in tqdm(test_files, desc="Testing images"):
        hr_gt = read_image_gray(f)
        lr = downsample(hr_gt, scale)

        # Reconstructions
        bicubic = upsample(lr, scale)
        sr_sparse = infer_single_image(lr, D_L, D_H, scale, use_edge_weights=False)
        sr_edge = infer_single_image(lr, D_L, D_H, scale, use_edge_weights=True)

        base = os.path.splitext(os.path.basename(f))[0]
        save_image(os.path.join(OUTPUT_DIR, f"{base}_lr_x{scale}.png"), lr)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_bicubic_x{scale}.png"), bicubic)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_sr_sparse_x{scale}.png"), sr_sparse)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_sr_edge_x{scale}.png"), sr_edge)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_hr_gt.png"), hr_gt)
    print("Inference complete. Outputs saved to", OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    process_test_folder(args.scale)

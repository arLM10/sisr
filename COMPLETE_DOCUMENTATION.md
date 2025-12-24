# Edge-Aware Dictionary Learning for Super-Resolution: Complete Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Core Concepts](#core-concepts)
4. [Algorithm Deep Dive](#algorithm-deep-dive)
5. [Code Architecture](#code-architecture)
6. [Step-by-Step Execution](#step-by-step-execution)
7. [Parameter Tuning](#parameter-tuning)
8. [Results Analysis](#results-analysis)

---

## 1. Project Overview

### What is Super-Resolution?
Super-Resolution (SR) is the process of reconstructing a high-resolution (HR) image from a low-resolution (LR) input. This is an ill-posed inverse problem because multiple HR images can produce the same LR image when downsampled.

### Why Edge-Aware Dictionary Learning?
Traditional sparse coding methods treat all image patches equally. However, edges and textures are crucial for visual quality. Edge-aware methods give special attention to preserving sharp boundaries and fine details.

### Project Goal
Develop a dictionary learning system that:
- Learns coupled dictionaries (LR and HR)
- Preserves edges and textures during reconstruction
- Achieves PSNR > 29 dB on test images
- Outperforms bicubic interpolation and standard sparse coding

---

## 2. Mathematical Foundations

### 2.1 Sparse Representation Theory

**Core Principle**: Any signal can be represented as a sparse linear combination of atoms from an overcomplete dictionary.

For a signal y ∈ ℝⁿ and dictionary D ∈ ℝⁿˣᵏ (where k > n):
```
y ≈ Dα
```
where α ∈ ℝᵏ is sparse (most elements are zero).

**Sparsity Constraint**: ||α||₀ ≤ s, where s << k

### 2.2 Dictionary Learning Problem

Given training signals Y = [y₁, y₂, ..., yₘ], find dictionary D and sparse codes A:

```
min_{D,A} ||Y - DA||²_F + λ||A||₁
```

Subject to: ||dⱼ||₂ = 1 for all dictionary atoms dⱼ

### 2.3 K-SVD Algorithm

K-SVD alternates between:
1. **Sparse Coding**: Fix D, solve for A using OMP
2. **Dictionary Update**: Fix A, update each atom using SVD

**Atom Update Step**:
For atom k:
- Compute error: E_k = Y - ∑_{j≠k} d_j a^T_j
- SVD: E_k = UΣV^T
- Update: d_k = u₁, a_k = σ₁v₁

### 2.4 Orthogonal Matching Pursuit (OMP)

OMP greedily selects dictionary atoms:

```
Algorithm OMP(D, y, s):
1. Initialize: residual r = y, support S = ∅
2. For i = 1 to s:
   a. Find best atom: k = argmax_j |⟨r, d_j⟩|
   b. Update support: S = S ∪ {k}
   c. Solve least squares: α_S = argmin ||y - D_S α||²
   d. Update residual: r = y - D_S α_S
3. Return α (sparse vector)
```

### 2.5 Edge-Aware Weighting

Standard OMP treats all atoms equally. Edge-aware OMP uses weights:

```
k = argmax_j w_j |⟨r, d_j⟩|
```

where w_j encourages edge-preserving atoms for edge-rich patches.

---

## 3. Core Concepts

### 3.1 Patch-Based Processing

**Why Patches?**
- Images have local structure and redundancy
- Small patches can be sparsely represented
- Enables parallel processing
- Reduces computational complexity

**Patch Extraction**:
- LR patches: 7×7 pixels
- HR patches: 18×18 pixels (7×2 + 4 padding)
- Overlapping extraction with step size 1

### 3.2 Coupled Dictionary Learning

**Problem**: Learn two dictionaries D_L and D_H such that:
- LR patches are sparse in D_L
- Corresponding HR patches are sparse in D_H
- Same sparse codes α work for both

**Mathematical Formulation**:
```
min_{D_L,D_H,A} ||Y_L - D_L A||²_F + ||Y_H - D_H A||²_F + λ||A||₁
```

### 3.3 Edge Detection and Weighting

**Gradient Magnitude**:
```python
gx = ∂I/∂x  # Sobel operator
gy = ∂I/∂y  # Sobel operator
magnitude = √(gx² + gy²)
```

**Edge Strength**: Average gradient magnitude in patch

**Weight Computation**:
```python
edge_factor = min(edge_strength × 3.0, 2.0)
texture_factor = min(patch_std × 2.0, 1.5)
weights = 1.0 + edge_factor × correlations + texture_factor × correlations
```

### 3.4 High-Frequency Component

**Why High-Frequency?**
- Contains edge and texture information
- Easier to learn than full HR patches
- Reduces low-frequency bias

**Computation**:
```
HF = HR - GaussianBlur(HR)
```

---

## 4. Algorithm Deep Dive

### 4.1 Training Phase

#### Step 1: Data Preparation
```python
for each HR training image:
    1. Crop to multiple of scale factor
    2. Downsample to create LR image
    3. Compute HF = HR - blur(HR)
    4. Extract overlapping patches
    5. Filter patches by edge strength
```

#### Step 2: Patch Selection
```python
for each LR patch:
    1. Compute gradient magnitude
    2. Calculate edge strength = mean(magnitude)
    3. If edge_strength > threshold:
        - Add to training set
        - Store corresponding HF patch
```

#### Step 3: Dictionary Learning
```python
# Learn LR dictionary
D_L = KSVD(LR_patches, n_atoms=256, sparsity=6)

# Compute sparse codes
A = OMP(D_L, LR_patches)

# Learn HR dictionary via least squares
D_H = solve(A^T × A, A^T × HF_patches^T)
```

### 4.2 Inference Phase

#### Step 1: Patch Extraction
```python
for each overlapping LR patch:
    1. Extract 7×7 patch
    2. Subtract mean (zero-center)
    3. Compute edge weights
```

#### Step 2: Edge-Aware Sparse Coding
```python
def weighted_OMP(D, y, weights):
    1. Scale dictionary: D_weighted = D × weights
    2. Run OMP on scaled dictionary
    3. Rescale coefficients: α = α_scaled × weights
    return α
```

#### Step 3: HR Reconstruction
```python
for each LR patch:
    1. α = weighted_OMP(D_L, patch, edge_weights)
    2. HR_patch = D_H × α
    3. Add back local mean
    4. Place in output image
```

#### Step 4: Patch Aggregation
```python
# Handle overlapping patches
for each pixel position:
    if multiple patches cover this pixel:
        final_value = average(all_patch_values)
```

#### Step 5: Back-Projection
```python
for iteration in range(back_proj_iters):
    1. Downsample current HR estimate
    2. Compute error = LR_input - downsampled
    3. Upsample error
    4. HR = HR + step_size × upsampled_error
    5. Clamp to [0, 1]
```

---

## 5. Code Architecture

### 5.1 File Structure
```
cvproject_final/
├── config.py          # All hyperparameters
├── utils.py           # Image I/O and processing utilities
├── ksvd.py            # K-SVD dictionary learning
├── train.py           # Training pipeline
├── sr_infer.py        # Inference pipeline
├── evaluate.py        # PSNR/SSIM evaluation
├── run.sh             # Main execution script
├── train512/          # Training images (300 images)
├── test_data20/       # Test images (20 images)
└── output/            # Results and dictionaries
```

### 5.2 Key Classes and Functions

#### config.py
```python
# Paths
HR_TRAIN_DIR = "path/to/training/images"
HR_TEST_DIR = "path/to/test/images"
OUTPUT_DIR = "path/to/output"

# Dictionary parameters
DICT_ATOMS = 256        # Number of dictionary atoms
SPARSITY = 6           # Target sparsity level
KSVD_ITERS = 8         # K-SVD iterations

# Patch parameters
LR_PATCH_SIZE = 7      # LR patch size
HR_PAD = 4             # HR patch padding
PATCH_STEP = 1         # Patch extraction step

# Edge filtering
EDGE_THRESHOLD = 0.3   # Minimum edge strength
```

#### utils.py
```python
def read_image_gray(path):
    """Load image as grayscale float32 [0,1]"""

def downsample(img, scale):
    """Downsample with anti-aliasing"""

def upsample(img, scale):
    """Bicubic upsampling"""

def gradient_magnitude(patch):
    """Compute Sobel gradient magnitude"""

def high_frequency(img):
    """Extract high-frequency component"""
```

#### ksvd.py
```python
class KSVDDictLearner:
    def __init__(self, n_atoms, sparsity, n_iter):
        """Initialize K-SVD parameters"""
    
    def fit(self, X):
        """Learn dictionary from training data"""
        for iteration in range(self.n_iter):
            # Sparse coding step
            Gamma = self._omp(D, X)
            
            # Dictionary update step
            for k in range(self.n_atoms):
                # Update atom k using SVD
    
    def _omp(self, D, X):
        """Orthogonal Matching Pursuit"""
```
#### train.py
```python
def build_training_patches(hr_files, scale):
    """Extract and filter training patches"""
    for each HR image:
        1. Load and preprocess image
        2. Create LR version by downsampling
        3. Compute high-frequency component
        4. Extract overlapping patches
        5. Filter by edge strength and texture
        6. Store LR and HF patch pairs

def train_for_scale(scale):
    """Complete training pipeline"""
    1. Load training images
    2. Extract patch pairs
    3. Normalize patches (zero-mean)
    4. Learn LR dictionary with K-SVD
    5. Compute sparse codes for all patches
    6. Learn HR dictionary via least squares
    7. Save both dictionaries
```

#### sr_infer.py
```python
def compute_edge_weights(lr_patch, D_L):
    """Compute edge-aware weights for dictionary atoms"""
    1. Calculate gradient magnitude
    2. Compute patch statistics (std, variance)
    3. Calculate atom correlations
    4. Combine edge and texture factors
    5. Return normalized weights

def weighted_omp(D, y, weights):
    """Edge-aware orthogonal matching pursuit"""
    if weights is None:
        return standard_OMP(D, y)
    else:
        1. Clamp weights to reasonable range
        2. Scale dictionary by weights
        3. Run OMP on scaled dictionary
        4. Rescale coefficients
        return sparse_coefficients

def infer_single_image(lr_img, D_L, D_H, scale, use_edge_weights):
    """Reconstruct HR image from LR input"""
    1. Initialize output arrays
    2. For each overlapping LR patch:
        a. Extract and normalize patch
        b. Compute edge weights (if enabled)
        c. Perform weighted sparse coding
        d. Reconstruct HR patch
        e. Add local mean back
    3. Aggregate overlapping patches
    4. Fill borders with bicubic interpolation
    5. Apply back-projection refinement
    6. Return final HR image
```

#### evaluate.py
```python
def evaluate(scale):
    """Compute PSNR and SSIM metrics"""
    for each test image:
        1. Load ground truth HR image
        2. Load reconstructed images (bicubic, sparse, edge-aware)
        3. Crop to same dimensions
        4. Compute PSNR and SSIM
        5. Save metrics to CSV files
```

---

## 6. Step-by-Step Execution

### 6.1 Training Process (train.py)

#### Phase 1: Image Loading and Preprocessing
```python
# Load all HR training images
hr_files = list_image_files(HR_TRAIN_DIR)  # Find all .png, .jpg files
print(f"Found {len(hr_files)} HR train images.")

for each image file:
    hr = read_image_gray(file)  # Load as grayscale [0,1]
    
    # Ensure dimensions are divisible by scale
    h, w = hr.shape
    hr = hr[:(h//scale)*scale, :(w//scale)*scale]  # Crop if needed
    
    # Create LR version
    lr = downsample(hr, scale)  # Anti-aliased downsampling
    
    # Compute high-frequency component
    hf = high_frequency(hr, ksize=5, sigma=1.0)  # hr - gaussian_blur(hr)
```

#### Phase 2: Patch Extraction and Filtering
```python
p_lr = LR_PATCH_SIZE  # 7
p_hr = LR_PATCH_SIZE * scale + HR_PAD  # 7*2 + 4 = 18

# Sliding window extraction
for i in range(0, lr.shape[0] - p_lr + 1, PATCH_STEP):  # Step = 1
    for j in range(0, lr.shape[1] - p_lr + 1, PATCH_STEP):
        # Extract LR patch
        lr_patch = lr[i:i+p_lr, j:j+p_lr]  # 7x7
        
        # Extract corresponding HR patch
        hi, hj = i * scale, j * scale
        hr_patch = hf[hi:hi+p_hr, hj:hj+p_hr]  # 18x18
        
        # Edge filtering
        mag = gradient_magnitude(lr_patch)  # Sobel gradients
        edge_strength = mag.mean()
        patch_variance = np.var(lr_patch)
        
        # Multi-criteria selection
        if edge_strength > EDGE_THRESHOLD or patch_variance > 0.01:
            lr_patches_list.append(lr_patch)
            hf_patches_list.append(hr_patch)
```

#### Phase 3: Dictionary Learning
```python
# Convert patches to vectors
X_lr = patches_to_vectors(lr_patches)  # (N, 49) for 7x7 patches
X_hf = patches_to_vectors(hf_patches)  # (N, 324) for 18x18 patches

# Zero-mean normalization
X_lr = X_lr - np.mean(X_lr, axis=1, keepdims=True)
X_hf = X_hf - np.mean(X_hf, axis=1, keepdims=True)

# Learn LR dictionary using K-SVD
ksvd_lr = KSVDDictLearner(n_atoms=256, sparsity=6, n_iter=8)
ksvd_lr.fit(X_lr)
D_L = ksvd_lr.D  # Shape: (49, 256)

# Compute sparse codes for all LR patches
Gamma = ksvd_lr._omp(D_L, X_lr)  # Shape: (256, N)

# Learn HR dictionary via least squares
# Solve: D_H * Gamma = X_hf^T
D_H, _, _, _ = np.linalg.lstsq(Gamma.T, X_hf, rcond=None)
D_H = D_H.T  # Shape: (324, 256)

# Normalize dictionary atoms
D_H = D_H / (np.linalg.norm(D_H, axis=0, keepdims=True) + 1e-12)

# Save dictionaries
np.save("D_L_x2.npy", D_L)
np.save("D_H_x2.npy", D_H)
```

### 6.2 Inference Process (sr_infer.py)

#### Phase 1: Dictionary Loading
```python
D_L = np.load("D_L_x2.npy")  # (49, 256)
D_H = np.load("D_H_x2.npy")  # (324, 256)
```

#### Phase 2: Test Image Processing
```python
for each test image:
    hr_gt = read_image_gray(test_file)  # Ground truth
    lr = downsample(hr_gt, scale)       # Create LR input
    
    # Three reconstruction methods:
    bicubic = upsample(lr, scale)                                    # Baseline
    sr_sparse = infer_single_image(lr, D_L, D_H, scale, False)      # No edge weights
    sr_edge = infer_single_image(lr, D_L, D_H, scale, True)         # With edge weights
```

#### Phase 3: Patch-by-Patch Reconstruction
```python
def infer_single_image(lr_img, D_L, D_H, scale, use_edge_weights):
    h_lr, w_lr = lr_img.shape
    target_shape = (h_lr * scale, w_lr * scale)
    
    # Initialize output arrays
    recon = np.zeros(target_shape, dtype=np.float32)
    weight = np.zeros(target_shape, dtype=np.float32)
    
    hr_patches = []
    
    # Process each overlapping patch
    for i in range(0, h_lr - p_lr + 1, PATCH_STEP):
        for j in range(0, w_lr - p_lr + 1, PATCH_STEP):
            # Extract LR patch
            lp = lr_img[i:i+p_lr, j:j+p_lr]  # 7x7
            vec = lp.reshape(-1) - np.mean(lp)  # Flatten and zero-mean
            
            # Compute edge-aware weights
            if use_edge_weights:
                weights = compute_edge_weights(lp, D_L)
            else:
                weights = None
            
            # Sparse coding
            alpha = weighted_omp(D_L, vec, n_nonzero_coefs=6, weights=weights)
            
            # HR reconstruction
            hr_vec = D_H.dot(alpha)  # Matrix-vector multiplication
            hr_patch = hr_vec.reshape(p_hr, p_hr)  # 18x18
            
            # Add back local mean
            mean_lr_up = np.mean(upsample(lp, scale))
            hr_patch = hr_patch + mean_lr_up
            
            hr_patches.append(hr_patch)
```

#### Phase 4: Patch Aggregation
```python
    # Aggregate overlapping patches
    idx = 0
    for i in range(0, h_lr - p_lr + 1, PATCH_STEP):
        for j in range(0, w_lr - p_lr + 1, PATCH_STEP):
            hi, hj = i * scale, j * scale
            patch = hr_patches[idx]
            
            # Add patch to reconstruction
            recon[hi:hi+p_hr, hj:hj+p_hr] += patch
            weight[hi:hi+p_hr, hj:hj+p_hr] += 1.0
            idx += 1
    
    # Normalize by overlap count
    mask = weight > 0
    recon[mask] /= weight[mask]
    
    # Fill uncovered borders
    bic = upsample(lr_img, scale)
    recon[~mask] = bic[~mask]
```

#### Phase 5: Back-Projection Refinement
```python
    # Iterative back-projection
    for iteration in range(BACKPROJECTION_ITERS):  # 12 iterations
        # Downsample current estimate
        down = downsample(recon, scale)
        
        # Compute error
        err = lr_img - down
        
        # Upsample error
        up_err = upsample(err, scale)
        
        # Adaptive step size
        step_size = 0.8 * (0.9 ** iteration)
        
        # Apply correction
        recon = recon + step_size * up_err
        recon = np.clip(recon, 0.0, 1.0)
    
    return recon
```

### 6.3 Edge Weight Computation Details

```python
def compute_edge_weights(lr_patch, D_L):
    # Compute gradient magnitude
    mag = gradient_magnitude(lr_patch)  # Sobel operators
    edge_strength = mag.mean()
    
    # Compute texture features
    patch_std = np.std(lr_patch)
    patch_var = np.var(lr_patch)
    
    # Base weights (uniform)
    weights = np.ones(D_L.shape[1])  # 256 atoms
    
    if edge_strength > 0.1:  # Edge threshold
        # Flatten and normalize patch
        lr_vec = lr_patch.reshape(-1)
        lr_vec = lr_vec - np.mean(lr_vec)
        
        # Compute correlations with dictionary atoms
        correlations = np.abs(D_L.T @ lr_vec)  # (256,)
        correlations = correlations / (np.linalg.norm(D_L, axis=0) * 
                                     np.linalg.norm(lr_vec) + 1e-12)
        
        # Compute scaling factors
        edge_factor = min(edge_strength * 3.0, 2.0)
        texture_factor = min(patch_std * 2.0, 1.5)
        
        # Combine factors
        weights = 1.0 + edge_factor * correlations + texture_factor * correlations
        
        # Clamp to reasonable range
        weights = np.clip(weights, 0.2, 5.0)
    
    return weights
```

### 6.4 Weighted OMP Algorithm

```python
def weighted_omp(D, y, n_nonzero_coefs, weights):
    if weights is None:
        # Standard OMP
        try:
            coef = orthogonal_mp(D, y.reshape(-1, 1), 
                               n_nonzero_coefs=n_nonzero_coefs)
            return coef.ravel()
        except:
            return np.zeros(D.shape[1])  # Fallback
    
    # Weighted OMP
    w = np.array(weights).clip(0.2, 5.0)  # Clamp weights
    
    # Scale dictionary atoms by weights
    D_weighted = D * w[None, :]  # Broadcasting
    
    try:
        # Solve with weighted dictionary
        coef_weighted = orthogonal_mp(D_weighted, y.reshape(-1, 1),
                                    n_nonzero_coefs=n_nonzero_coefs)
        # Rescale coefficients
        coef = coef_weighted.ravel() * w
        return coef
    except:
        # Fallback to standard OMP
        try:
            coef = orthogonal_mp(D, y.reshape(-1, 1),
                               n_nonzero_coefs=n_nonzero_coefs)
            return coef.ravel()
        except:
            return np.zeros(D.shape[1])
```

---

## 7. Parameter Tuning Guide

### 7.1 Dictionary Parameters

#### DICT_ATOMS (Number of Dictionary Atoms)
- **Range**: 128-512
- **Current**: 256
- **Effect**: 
  - Higher → Better representation, more computation
  - Lower → Faster training, potential underfitting
- **Tuning**: Start with 256, increase if PSNR plateaus

#### SPARSITY (Target Non-zeros in OMP)
- **Range**: 3-10
- **Current**: 6
- **Effect**:
  - Higher → More atoms used, richer representation
  - Lower → Sparser codes, faster inference
- **Tuning**: Should be ~2-5% of DICT_ATOMS

#### KSVD_ITERS (K-SVD Iterations)
- **Range**: 5-15
- **Current**: 8
- **Effect**:
  - Higher → Better convergence, more computation
  - Lower → Faster training, potential underfitting
- **Tuning**: Monitor reconstruction error convergence

### 7.2 Patch Parameters

#### LR_PATCH_SIZE (Low-Resolution Patch Size)
- **Range**: 5-9 (odd numbers)
- **Current**: 7
- **Effect**:
  - Larger → More context, more computation
  - Smaller → Less context, faster processing
- **Tuning**: 7x7 is good balance for 2x upscaling

#### HR_PAD (High-Resolution Padding)
- **Range**: 2-6
- **Current**: 4
- **Effect**:
  - Larger → More HR context, better reconstruction
  - Smaller → Less computation, potential artifacts
- **Formula**: HR_patch_size = LR_PATCH_SIZE × scale + HR_PAD

#### PATCH_STEP (Patch Extraction Step)
- **Range**: 1-3
- **Current**: 1
- **Effect**:
  - 1 → Maximum overlap, best quality, slow
  - 2+ → Less overlap, faster, potential artifacts
- **Tuning**: Use 1 for best quality, 2 for speed

### 7.3 Edge Parameters

#### EDGE_THRESHOLD (Minimum Edge Strength)
- **Range**: 0.1-1.0
- **Current**: 0.3
- **Effect**:
  - Lower → More patches selected, diverse training
  - Higher → Only strong edges, focused learning
- **Tuning**: Adjust based on training data characteristics

#### Weight Clipping (in weighted_omp)
- **Range**: [0.1-0.5, 3.0-10.0]
- **Current**: [0.2, 5.0]
- **Effect**:
  - Wider range → More aggressive weighting
  - Narrower range → More conservative weighting
- **Tuning**: Start conservative, increase if edge preservation needed

### 7.4 Training Parameters

#### MAX_PATCHES (Maximum Training Patches)
- **Range**: 50K-500K or None
- **Current**: None (unlimited)
- **Effect**:
  - Higher → Better dictionary, more memory/time
  - Lower → Faster training, potential underfitting
- **Tuning**: Use 100K-300K for good balance

#### BACKPROJECTION_ITERS (Back-projection Iterations)
- **Range**: 5-20
- **Current**: 12
- **Effect**:
  - Higher → Better LR consistency, more computation
  - Lower → Faster inference, potential artifacts
- **Tuning**: 8-12 is usually sufficient

---

## 8. Results Analysis

### 8.1 Quality Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
```
PSNR = 20 × log₁₀(MAX_I / √MSE)
```
- **Range**: 20-40 dB (higher is better)
- **Target**: >29 dB for good quality
- **Interpretation**:
  - 25-30 dB: Acceptable quality
  - 30-35 dB: Good quality
  - 35+ dB: Excellent quality

#### SSIM (Structural Similarity Index)
```
SSIM = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))
```
- **Range**: 0-1 (higher is better)
- **Target**: >0.85 for good quality
- **Interpretation**:
  - 0.7-0.8: Acceptable similarity
  - 0.8-0.9: Good similarity
  - 0.9+: Excellent similarity

### 8.2 Expected Performance

#### Bicubic Baseline
- **PSNR**: ~24-26 dB
- **SSIM**: ~0.75-0.85
- **Characteristics**: Smooth but blurry

#### Standard Sparse Coding
- **PSNR**: ~26-28 dB
- **SSIM**: ~0.80-0.88
- **Characteristics**: Sharper than bicubic, some artifacts

#### Edge-Aware Method (This Project)
- **PSNR**: ~28-32 dB (target >29)
- **SSIM**: ~0.85-0.92
- **Characteristics**: Sharp edges, preserved textures

### 8.3 Troubleshooting Common Issues

#### Low PSNR (<27 dB)
**Possible Causes**:
- Dictionary too small (increase DICT_ATOMS)
- Insufficient training data (check MAX_PATCHES)
- Poor patch selection (lower EDGE_THRESHOLD)
- Inadequate sparsity (increase SPARSITY)

**Solutions**:
1. Increase DICT_ATOMS to 512
2. Ensure diverse training images
3. Lower EDGE_THRESHOLD to 0.2
4. Increase SPARSITY to 8

#### "Pursuit Ended Prematurely" Warning
**Cause**: Linear dependence in dictionary
**Solutions**:
1. Reduce DICT_ATOMS
2. Increase training data diversity
3. Add regularization to K-SVD

#### No Difference Between Methods
**Cause**: Edge weighting not working
**Solutions**:
1. Check weight computation
2. Verify edge detection
3. Adjust weight clipping range

#### Artifacts in Results
**Possible Causes**:
- Large PATCH_STEP (use 1)
- Insufficient back-projection (increase iterations)
- Poor patch aggregation

**Solutions**:
1. Set PATCH_STEP = 1
2. Increase BACKPROJECTION_ITERS
3. Check patch overlap handling

### 8.4 Performance Optimization

#### Memory Usage
- **Training**: ~2-4 GB for 300K patches
- **Inference**: ~500 MB per image
- **Optimization**: Process images in batches

#### Computation Time
- **Training**: 10-30 minutes (depends on data size)
- **Inference**: 30-60 seconds per image
- **Optimization**: Use N_JOBS = -1 for parallel processing

#### Storage Requirements
- **Dictionaries**: ~5 MB per scale
- **Training Images**: ~500 MB (300 images)
- **Results**: ~50 MB per test set

---

## 9. Advanced Topics

### 9.1 Multi-Scale Training
Train separate dictionaries for different scales (2x, 3x, 4x):
```bash
python train.py --scale 2
python train.py --scale 3
python train.py --scale 4
```

### 9.2 Color Image Processing
Extend to color images by:
1. Converting to YUV color space
2. Processing Y channel with current method
3. Upsampling U,V channels with bicubic
4. Converting back to RGB

### 9.3 Real-Time Optimization
For faster inference:
1. Reduce dictionary size (128 atoms)
2. Increase patch step (2-3)
3. Reduce back-projection iterations (5-8)
4. Use GPU acceleration

### 9.4 Quality Enhancement
For better quality:
1. Larger dictionaries (512-1024 atoms)
2. Multi-scale dictionaries
3. Adaptive sparsity based on patch content
4. Non-local similarity constraints

---

## 10. Conclusion

This edge-aware dictionary learning system represents a sophisticated approach to single-image super-resolution. By combining sparse representation theory with edge-aware processing, it achieves superior reconstruction quality compared to traditional methods.

**Key Innovations**:
1. **Edge-aware weighting** in sparse coding
2. **Coupled dictionary learning** for LR-HR correspondence
3. **Multi-criteria patch selection** for diverse training
4. **Robust implementation** with comprehensive error handling

**Expected Outcomes**:
- PSNR improvement of 3-5 dB over bicubic
- Better edge and texture preservation
- Competitive performance with modern methods
- Solid foundation for further research

The system is designed to be both educational and practical, providing insights into the mathematical foundations while delivering real improvements in image quality.

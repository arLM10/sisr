# Edge-Aware Dictionary Learning: Deep Mathematical Foundations

## Table of Contents
1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Core Concepts](#2-core-concepts)
3. [Algorithm Deep Dive](#3-algorithm-deep-dive)
4. [Step-by-Step Execution](#4-step-by-step-execution)

---

## 1. Mathematical Foundations

### 1.1 Sparse Representation Theory - The Foundation

#### 1.1.1 The Fundamental Problem
In classical signal processing, we represent signals using orthogonal bases (Fourier, wavelets). However, natural signals often have structure that can't be efficiently captured by fixed bases. Sparse representation theory addresses this by using **overcomplete dictionaries**.

**Definition**: A dictionary D ∈ ℝⁿˣᵏ is overcomplete if k > n, meaning we have more atoms (columns) than the signal dimension.

**Core Principle**: Any signal y ∈ ℝⁿ can be represented as:
```
y = Dα + ε
```
where:
- D ∈ ℝⁿˣᵏ is the dictionary matrix
- α ∈ ℝᵏ is the sparse coefficient vector
- ε is the approximation error
- ||α||₀ ≤ s << k (sparsity constraint)

#### 1.1.2 Why Sparsity Works - Information Theory Perspective

**Kolmogorov Complexity**: The shortest description of a signal is its most compressed representation. Natural images have structure and redundancy, making them compressible.

**Manifold Hypothesis**: High-dimensional natural signals (image patches) lie on or near low-dimensional manifolds. Sparse coding finds these intrinsic coordinates.

**Mathematical Justification**:
If signals lie on a union of subspaces, each of dimension d << n, then:
- We need only d coefficients to represent each signal
- Total degrees of freedom: k·d (much less than n·k for dense representation)
- This explains why sparse representation is so effective

#### 1.1.3 The Sparse Coding Problem - Optimization Perspective

**P0 Problem (Ideal but NP-hard)**:
```
min ||α||₀ subject to ||y - Dα||₂ ≤ ε
```

**P1 Relaxation (Convex approximation)**:
```
min ||α||₁ subject to ||y - Dα||₂ ≤ ε
```

**Lagrangian Form**:
```
min (1/2)||y - Dα||₂² + λ||α||₁
```

**Why L1 Works**: The L1 norm is the tightest convex relaxation of L0. Geometrically, the L1 ball has corners at sparse points, encouraging sparse solutions.

#### 1.1.4 Restricted Isometry Property (RIP)

**Definition**: A matrix D satisfies RIP of order s with constant δₛ if:
```
(1 - δₛ)||α||₂² ≤ ||Dα||₂² ≤ (1 + δₛ)||α||₂²
```
for all s-sparse vectors α.

**Significance**: If δ₂ₛ < √2 - 1 ≈ 0.414, then OMP recovers the sparsest solution exactly.

**Intuition**: RIP ensures that sparse vectors are not mapped to the null space - the dictionary preserves the geometry of sparse signals.

### 1.2 Dictionary Learning Theory

#### 1.2.1 The Bi-convex Problem
Dictionary learning solves:
```
min_{D,A} ||Y - DA||²_F + λ||A||₁
subject to ||dⱼ||₂ = 1 ∀j
```

This is **bi-convex**:
- Convex in D when A is fixed
- Convex in A when D is fixed
- Non-convex jointly

#### 1.2.2 Alternating Minimization Framework

**Block Coordinate Descent**: Alternate between:
1. **Sparse Coding**: A^(t+1) = argmin_A ||Y - D^(t)A||²_F + λ||A||₁
2. **Dictionary Update**: D^(t+1) = argmin_D ||Y - DA^(t+1)||²_F s.t. ||dⱼ||₂ = 1

**Convergence Theory**: Under certain conditions (coherence bounds, RIP), this converges to a local minimum.

#### 1.2.3 K-SVD Algorithm - Detailed Mathematical Analysis

**Atom-by-Atom Update**: Instead of updating the entire dictionary, K-SVD updates one atom at a time.

For atom k:
1. **Error Computation**: 
   ```
   E_k = Y - ∑_{j≠k} d_j a^T_j = Y - D_{-k}A_{-k}
   ```

2. **Restriction to Active Set**: 
   Only consider samples that use atom k:
   ```
   Ω_k = {i : A(k,i) ≠ 0}
   E^R_k = E_k[:, Ω_k]
   ```

3. **SVD Decomposition**:
   ```
   E^R_k = UΣV^T
   ```

4. **Update**:
   ```
   d_k = u₁ (first left singular vector)
   A(k, Ω_k) = σ₁v₁^T (scaled first right singular vector)
   ```

**Why SVD?**: SVD finds the best rank-1 approximation to E^R_k, which is exactly what we need for updating one atom and its coefficients.

**Geometric Interpretation**: We're finding the direction (d_k) that best explains the residual error for samples using atom k.

### 1.3 Orthogonal Matching Pursuit - Deep Dive

#### 1.3.1 The Greedy Strategy
OMP builds the support set S iteratively by selecting the atom most correlated with the current residual.

**Algorithm**:
```
Initialize: r₀ = y, S₀ = ∅, t = 0
While ||rₜ||₂ > ε and |Sₜ| < s:
    1. λₜ₊₁ = argmax_j |⟨rₜ, dⱼ⟩|
    2. Sₜ₊₁ = Sₜ ∪ {λₜ₊₁}
    3. αₛₜ₊₁ = argmin ||y - D_Sₜ₊₁ α||₂²
    4. rₜ₊₁ = y - D_Sₜ₊₁ αₛₜ₊₁
    5. t = t + 1
```

#### 1.3.2 Geometric Interpretation
- **Step 1**: Find the dictionary atom most aligned with the residual
- **Step 3**: Project y onto span(D_S) - orthogonal projection
- **Step 4**: Compute new residual - orthogonal to span(D_S)

**Key Insight**: Each iteration reduces the residual in the direction most correlated with the dictionary, ensuring orthogonality.

#### 1.3.3 Convergence Analysis

**Exact Recovery Conditions**: If the true sparse representation has support S* and:
```
max_{j∉S*} |⟨D_S*, dⱼ⟩| < 1/(2s-1)
```
then OMP recovers S* exactly in s iterations.

**Approximation Bounds**: For approximately sparse signals:
```
||y - D α_OMP||₂ ≤ C₁ σₛ(α*)/√s
```
where σₛ(α*) is the best s-term approximation error of the true coefficients.

### 1.4 Edge-Aware Sparse Coding - Novel Contribution

#### 1.4.1 Motivation from Human Visual System
The human visual system is particularly sensitive to edges and discontinuities. Traditional sparse coding treats all atoms equally, but for image reconstruction, edge-preserving atoms should be prioritized.

#### 1.4.2 Weighted Sparse Coding Formulation
Instead of standard sparse coding:
```
min ||y - Dα||₂² + λ||α||₁
```

We solve:
```
min ||y - Dα||₂² + λ||Wα||₁
```
where W = diag(w₁, w₂, ..., wₖ) contains atom-specific weights.

#### 1.4.3 Edge-Aware Weight Design

**Gradient-Based Edge Detection**:
```
∇I = [∂I/∂x, ∂I/∂y]
||∇I|| = √((∂I/∂x)² + (∂I/∂y)²)
```

**Edge Strength Measure**:
```
E(patch) = (1/|patch|) ∑ ||∇I(x,y)||
```

**Atom Correlation**:
For patch p and dictionary D:
```
corr_j = |⟨p̃, d_j⟩| / (||p̃|| ||d_j||)
```
where p̃ is the zero-mean patch.

**Weight Computation**:
```
w_j = 1 + α·E(patch)·corr_j + β·texture_factor·corr_j
```

**Intuition**: 
- High edge strength → prioritize correlated atoms
- High correlation → this atom can represent the patch well
- Combined → edge-preserving atoms get higher priority

#### 1.4.4 Weighted OMP Algorithm

**Modified Selection Step**:
Instead of: λₜ₊₁ = argmax_j |⟨rₜ, dⱼ⟩|
Use: λₜ₊₁ = argmax_j wⱼ|⟨rₜ, dⱼ⟩|

**Implementation via Dictionary Scaling**:
```
D̃ = D · diag(w₁, w₂, ..., wₖ)
α̃ = OMP(D̃, y)
α = α̃ .* w (element-wise multiplication)
```

**Mathematical Justification**: This is equivalent to solving the weighted L1 problem with a diagonal weight matrix.

### 1.5 Coupled Dictionary Learning for Super-Resolution

#### 1.5.1 The Super-Resolution Problem
Given low-resolution image I_L, find high-resolution image I_H such that:
```
I_L = (I_H * h) ↓ s + n
```
where:
- h is the blur kernel
- ↓s is downsampling by factor s
- n is noise

This is **severely ill-posed**: infinitely many I_H can produce the same I_L.

#### 1.5.2 Sparse Prior for Regularization
**Assumption**: Corresponding LR and HR patches have the same sparse representation in their respective dictionaries.

**Mathematical Formulation**:
```
y_L = D_L α + ε_L
y_H = D_H α + ε_H
```

Same α for both representations!

#### 1.5.3 Joint Dictionary Learning Problem
```
min_{D_L,D_H,A} ||Y_L - D_L A||²_F + ||Y_H - D_H A||²_F + λ||A||₁
subject to ||d^L_j||₂ = 1, ||d^H_j||₂ = 1 ∀j
```

**Two-Stage Solution**:
1. Learn D_L from LR patches using K-SVD
2. Compute sparse codes: A = OMP(D_L, Y_L)
3. Learn D_H by solving: min_{D_H} ||Y_H - D_H A||²_F

**Stage 3 Solution**: This is a least squares problem:
```
D_H = Y_H A^T (AA^T)^(-1)
```

#### 1.5.4 High-Frequency Component Learning
Instead of learning full HR patches, we learn the high-frequency component:
```
Y_HF = Y_H - blur(Y_H)
```

**Advantages**:
- Removes low-frequency bias
- Focuses on edges and textures
- Easier to learn (less variation)
- Better generalization

**Reconstruction**:
```
ŷ_H = D_H α + mean(upsample(y_L))
```

### 1.6 Information Theory Perspective

#### 1.6.1 Rate-Distortion Theory
**Rate-Distortion Function**: R(D) = min I(X;Y) subject to E[d(X,Y)] ≤ D

For sparse coding:
- **Rate**: Number of non-zero coefficients × log(quantization levels)
- **Distortion**: ||y - Dα||₂²

**Optimal Allocation**: Sparse coding naturally allocates bits where they matter most (high-energy coefficients).

#### 1.6.2 Mutual Information in Coupled Dictionaries
The shared sparse code α contains the mutual information between LR and HR patches:
```
I(Y_L; Y_H) ≈ I(α_L; α_H) = H(α) (since α_L = α_H = α)
```

**Interpretation**: The sparse code captures the essential information shared between resolutions.

#### 1.6.3 Edge Information and Entropy
Edges carry most of the visual information in images:
```
H(image) ≈ H(edges) + H(smooth regions)
```

Since H(smooth regions) is low, preserving edges preserves most information.

**Edge-Aware Coding**: By prioritizing edge-preserving atoms, we maximize information preservation with minimal coefficients.

---

## 2. Core Concepts

### 2.1 Patch-Based Image Processing - Theoretical Foundation

#### 2.1.1 Why Patches Work - Statistical Perspective

**Local Stationarity**: Natural images are globally non-stationary but locally stationary. Small patches can be modeled with simpler statistics.

**Markov Property**: Pixel dependencies decay rapidly with distance. For most natural images:
```
P(I(x,y) | I(neighborhood)) ≈ P(I(x,y) | I(local patch))
```

**Curse of Dimensionality**: Full images live in extremely high-dimensional spaces. Patches reduce dimensionality while preserving local structure.

#### 2.1.2 Patch Size Selection - Information Theory

**Too Small**: Insufficient context, poor sparse representation
**Too Large**: Curse of dimensionality, computational complexity

**Optimal Size**: Balance between:
- **Mutual Information**: I(patch, context)
- **Computational Complexity**: O(n³) for n×n patches
- **Dictionary Size**: Exponential growth with patch size

**Empirical Finding**: 5×5 to 9×9 patches work best for natural images at typical resolutions.

#### 2.1.3 Overlapping vs Non-overlapping Patches

**Non-overlapping**:
- Pros: No redundancy, faster processing
- Cons: Block artifacts, discontinuities

**Overlapping**:
- Pros: Smooth reconstruction, better quality
- Cons: Redundancy, aggregation complexity

**Aggregation Theory**: For overlapping patches with reconstruction r_i at position i:
```
final(x,y) = ∑ᵢ w_i(x,y) r_i(x,y) / ∑ᵢ w_i(x,y)
```

**Optimal Weights**: Uniform weighting minimizes reconstruction variance under Gaussian noise.

### 2.2 Multi-Resolution Analysis and Wavelets Connection

#### 2.2.1 Relationship to Wavelet Theory
Dictionary learning can be seen as **adaptive wavelet construction**:
- Wavelets: Fixed basis functions at multiple scales
- Dictionary atoms: Learned basis functions adapted to data

**Advantage**: Dictionary atoms adapt to image content, while wavelets are fixed.

#### 2.2.2 Scale-Space Theory
**Gaussian Scale-Space**: I(x,y,σ) = I(x,y) * G(x,y,σ)

**Connection to Super-Resolution**:
- LR image ≈ HR image at larger scale
- Super-resolution ≈ inverse scale-space operation
- Dictionary learning finds the inverse mapping

#### 2.2.3 Frequency Domain Analysis
**Fourier Perspective**: 
- LR patches have limited bandwidth
- HR patches have extended bandwidth
- Dictionary learning finds the bandwidth extension mapping

**Spectral Coherence**: Corresponding LR-HR patches should have coherent spectral structure in their overlapping frequency bands.

### 2.3 Edge Detection and Analysis - Mathematical Framework

#### 2.3.1 Differential Geometry of Images
Images can be viewed as 2D manifolds embedded in 3D space (x,y,intensity).

**First Fundamental Form**:
```
ds² = (1 + I_x²)dx² + 2I_x I_y dxdy + (1 + I_y²)dy²
```

**Gaussian Curvature**:
```
K = (I_xx I_yy - I_xy²) / (1 + I_x² + I_y²)²
```

**Mean Curvature**:
```
H = (I_xx(1 + I_y²) - 2I_xy I_x I_y + I_yy(1 + I_x²)) / (2(1 + I_x² + I_y²)^(3/2))
```

**Edge Interpretation**: Edges correspond to high curvature regions.

#### 2.3.2 Scale-Space Edge Detection
**Canny Edge Detection**: Multi-scale approach using Gaussian derivatives:
```
G_σ(x,y) = (1/2πσ²) exp(-(x² + y²)/2σ²)
∇(I * G_σ) = I * ∇G_σ
```

**Edge Strength**: ||∇(I * G_σ)||
**Edge Direction**: arctan(∂(I * G_σ)/∂y, ∂(I * G_σ)/∂x)

#### 2.3.3 Structure Tensor Analysis
**Structure Tensor**:
```
J = [I_x²    I_x I_y]
    [I_x I_y  I_y²  ]
```

**Eigenvalue Analysis**:
- λ₁ >> λ₂ ≈ 0: Edge (dominant direction)
- λ₁ ≈ λ₂ >> 0: Corner (no dominant direction)
- λ₁ ≈ λ₂ ≈ 0: Flat region

**Coherence Measure**:
```
coherence = (λ₁ - λ₂)² / (λ₁ + λ₂)²
```

**Edge-Aware Weighting**: Use coherence and eigenvalues to determine edge importance.

### 2.4 Texture Analysis and Characterization

#### 2.4.1 Local Binary Patterns (LBP)
**Definition**: For each pixel, compare with neighbors:
```
LBP(x,y) = ∑ᵢ s(I(xᵢ,yᵢ) - I(x,y)) 2ⁱ
```
where s(z) = 1 if z ≥ 0, else 0.

**Texture Characterization**: LBP histogram describes local texture patterns.

#### 2.4.2 Gray-Level Co-occurrence Matrix (GLCM)
**Definition**: P(i,j|d,θ) = probability that pixels separated by distance d in direction θ have intensities i and j.

**Texture Features**:
- **Contrast**: ∑ᵢ,ⱼ (i-j)² P(i,j)
- **Homogeneity**: ∑ᵢ,ⱼ P(i,j)/(1 + |i-j|)
- **Energy**: ∑ᵢ,ⱼ P(i,j)²

#### 2.4.3 Gabor Filter Analysis
**Gabor Function**:
```
g(x,y) = exp(-(x'²/σ_x² + y'²/σ_y²)/2) cos(2πfx' + φ)
```
where (x',y') are rotated coordinates.

**Multi-scale, Multi-orientation**: Bank of Gabor filters captures texture at different scales and orientations.

**Connection to Dictionary Learning**: Dictionary atoms often resemble Gabor functions, but are adapted to specific image content.

### 2.5 Perceptual Quality Metrics - Beyond PSNR

#### 2.5.1 Human Visual System (HVS) Modeling
**Contrast Sensitivity Function**: HVS sensitivity varies with spatial frequency:
```
CSF(f) = a f exp(-b f) √(1 + c f)
```

**Just Noticeable Difference (JND)**: Minimum perceivable difference varies with local image properties.

#### 2.5.2 Structural Similarity (SSIM) - Deep Analysis
**SSIM Formula**:
```
SSIM(x,y) = (2μ_x μ_y + c₁)(2σ_xy + c₂) / ((μ_x² + μ_y² + c₁)(σ_x² + σ_y² + c₂))
```

**Three Components**:
1. **Luminance**: l(x,y) = (2μ_x μ_y + c₁)/(μ_x² + μ_y² + c₁)
2. **Contrast**: c(x,y) = (2σ_x σ_y + c₂)/(σ_x² + σ_y² + c₂)
3. **Structure**: s(x,y) = (σ_xy + c₃)/(σ_x σ_y + c₃)

**Interpretation**: SSIM measures structural information preservation, which correlates better with human perception than MSE.

#### 2.5.3 Multi-Scale SSIM (MS-SSIM)
**Motivation**: Human vision is sensitive to structures at multiple scales.

**Algorithm**:
1. Compute SSIM at original resolution
2. Downsample both images
3. Repeat until minimum resolution
4. Combine: MS-SSIM = ∏ᵢ SSIM(i)^αᵢ

### 2.6 Optimization Theory for Dictionary Learning

#### 2.6.1 Non-Convex Optimization Landscape
Dictionary learning is non-convex, but has special structure:

**Local Minima**: Many local minima exist, but most are "good" (lead to similar reconstruction quality).

**Saddle Points**: More problematic than local minima. Modern optimization avoids saddle points.

**Global Minimum**: Typically not unique due to permutation and scaling ambiguities.

#### 2.6.2 Convergence Analysis
**Block Coordinate Descent**: Alternating minimization between D and A.

**Convergence Rate**: Under certain conditions:
```
f(D^(t), A^(t)) - f* ≤ O(1/t)
```

**Practical Convergence**: Usually converges in 10-20 iterations for most problems.

#### 2.6.3 Initialization Strategies
**Random Initialization**: Sample from training data
**K-means++**: Better initialization for clustering-based methods
**SVD Initialization**: Use SVD of data matrix

**Impact**: Good initialization can reduce convergence time by 2-3x.

### 2.7 Generalization Theory

#### 2.7.1 Statistical Learning Theory
**Generalization Bound**: For dictionary learning with n samples, k atoms, sparsity s:
```
E[test_error] ≤ training_error + O(√(ks log(n)/n))
```

**Interpretation**: Generalization improves with more data, but degrades with dictionary size and sparsity.

#### 2.7.2 Bias-Variance Tradeoff
**Bias**: Error from approximating complex relationships with simple model
**Variance**: Error from sensitivity to training data

**Dictionary Size**:
- Small dictionary: High bias, low variance
- Large dictionary: Low bias, high variance
- Optimal size: Minimize bias + variance

#### 2.7.3 Cross-Validation for Model Selection
**K-fold CV**: Split data into K folds, train on K-1, test on 1.

**Hyperparameter Selection**:
- Dictionary size: 64, 128, 256, 512
- Sparsity level: 3, 6, 9, 12
- Regularization: λ ∈ [0.01, 0.1, 1.0]

**Nested CV**: Outer loop for performance estimation, inner loop for hyperparameter selection.

This completes the first part focusing on deep mathematical foundations and core concepts. The content is now much more detailed and theoretical, explaining the "why" and "how" behind every mathematical concept.
---

## 3. Algorithm Deep Dive

### 3.1 K-SVD Algorithm - Complete Mathematical Analysis

#### 3.1.1 The Rank-1 Update Problem
At each iteration, K-SVD updates one dictionary atom. This is equivalent to solving:
```
min_{d_k, a_k} ||E_k - d_k a_k^T||_F^2
```
where E_k is the error matrix when atom k is removed.

**Matrix Factorization Perspective**: We seek the best rank-1 approximation to E_k.

**SVD Solution**: If E_k = UΣV^T, then:
- Optimal d_k = u_1 (first left singular vector)
- Optimal a_k = σ_1 v_1 (scaled first right singular vector)

**Geometric Interpretation**: 
- u_1 is the direction of maximum variance in the column space
- v_1 is the direction of maximum variance in the row space
- σ_1 is the amount of variance explained

#### 3.1.2 Restricted SVD for Sparsity Preservation
**Problem**: Standard SVD might change the sparsity pattern of coefficients.

**Solution**: Restrict SVD to samples that actually use atom k:
```
Ω_k = {i : A(k,i) ≠ 0}
E_k^R = E_k[:, Ω_k]
```

**Restricted Update**:
1. Compute SVD: E_k^R = UΣV^T
2. Update: d_k = u_1
3. Update coefficients: A(k, Ω_k) = σ_1 v_1^T

**Sparsity Preservation**: This ensures that zeros remain zeros, maintaining the sparsity pattern.

#### 3.1.3 Convergence Analysis of K-SVD

**Objective Function**: F(D,A) = ||Y - DA||_F^2

**Monotonic Decrease**: Each atom update decreases the objective:
```
F(D^{t+1}, A^{t+1}) ≤ F(D^t, A^t)
```

**Convergence Proof Sketch**:
1. F is bounded below by 0
2. Each update decreases F by at least ε > 0 (unless at optimum)
3. Therefore, F converges to a limit

**Local Minimum**: K-SVD converges to a local minimum, not necessarily global.

**Practical Convergence**: Usually 5-15 iterations sufficient for most applications.

#### 3.1.4 Dictionary Coherence and Recovery Guarantees

**Mutual Coherence**: 
```
μ(D) = max_{i≠j} |⟨d_i, d_j⟩|
```

**Recovery Guarantee**: If the true sparsity level s satisfies:
```
s < (1 + 1/μ(D))/2
```
then OMP recovers the true sparse representation exactly.

**Coherence Control**: K-SVD doesn't explicitly control coherence, but empirically produces low-coherence dictionaries.

### 3.2 Orthogonal Matching Pursuit - Advanced Analysis

#### 3.2.1 Geometric Interpretation in High Dimensions
**Residual Orthogonality**: After t iterations, the residual r_t is orthogonal to span(D_S_t):
```
⟨r_t, d_j⟩ = 0 ∀j ∈ S_t
```

**Projection Interpretation**: Each OMP iteration projects the signal onto a higher-dimensional subspace:
```
P_t = D_S_t (D_S_t^T D_S_t)^{-1} D_S_t^T
y_t = P_t y
r_t = (I - P_t) y
```

**Gram-Schmidt Process**: OMP implicitly performs Gram-Schmidt orthogonalization of selected atoms.

#### 3.2.2 Exact Recovery Conditions - Detailed Analysis

**Exact Recovery Criterion (ERC)**:
For exact recovery of support S*, we need:
```
max_{j∉S*} ||D_S*^T d_j||_1 < 1
```

**Mutual Coherence Condition**:
If μ(D) is the mutual coherence, then exact recovery is guaranteed if:
```
||α*||_0 < (1 + 1/μ(D))/2
```

**Restricted Isometry Property (RIP)**:
If D satisfies RIP with δ_{2s} < √2 - 1, then OMP recovers s-sparse signals exactly.

**Noise Robustness**: Under noise ||e||_2 ≤ ε, OMP produces:
```
||α - α*||_2 ≤ C ε
```
where C depends on the RIP constant.

#### 3.2.3 Computational Complexity Analysis

**Naive Implementation**: O(knm) per iteration, where:
- k = dictionary size
- n = signal dimension  
- m = number of signals

**Optimized Implementation**:
1. **Precompute Gram Matrix**: G = D^T D (O(k^2 n) once)
2. **Cholesky Updates**: Update Cholesky factorization incrementally
3. **Complexity**: O(t^2 k + t^3) per iteration, where t is current sparsity

**Memory Requirements**: O(k^2) for Gram matrix, O(tk) for active set.

#### 3.2.4 Batch OMP for Multiple Signals

**Simultaneous OMP (SOMP)**:
Select atoms that are most correlated with multiple signals simultaneously:
```
λ_{t+1} = argmax_j ||D_j^T R_t||_F
```
where R_t is the residual matrix for all signals.

**Advantages**:
- Shared sparsity patterns
- Better statistical stability
- Computational efficiency

### 3.3 Edge-Aware Weighting - Theoretical Foundation

#### 3.3.1 Perceptual Weighting Theory
**Weber-Fechner Law**: Human perception follows logarithmic response:
```
S = k log(I/I_0)
```
where S is perceived intensity, I is actual intensity.

**Contrast Sensitivity**: Human vision is most sensitive to contrast changes, especially at edges.

**Weighting Justification**: Edge-aware weights align sparse coding with human perceptual priorities.

#### 3.3.2 Information-Theoretic Weighting

**Mutual Information**: Between patch content and reconstruction quality:
```
I(patch_type; quality) = H(quality) - H(quality|patch_type)
```

**Edge Information**: Edges carry disproportionate visual information:
```
H(image) ≈ α H(edges) + (1-α) H(smooth_regions)
```
where α >> (1-α) for natural images.

**Optimal Weighting**: Weights should be proportional to information content:
```
w_j ∝ I(atom_j; edge_content)
```

#### 3.3.3 Bayesian Interpretation of Weights

**Prior Distribution**: Assume different priors for different atom types:
```
p(α_j) ∝ exp(-λ_j |α_j|)
```

**Edge-Adaptive Priors**: For edge-rich patches:
```
λ_j = λ_0 / w_j
```
where w_j > 1 for edge-preserving atoms.

**MAP Estimation**: Weighted sparse coding corresponds to MAP estimation with adaptive priors.

#### 3.3.4 Multi-Scale Edge Analysis

**Gaussian Scale-Space**: Analyze edges at multiple scales:
```
L(x,y,σ) = I(x,y) * G(x,y,σ)
∇L(x,y,σ) = I(x,y) * ∇G(x,y,σ)
```

**Scale-Space Edge Strength**:
```
E(x,y) = ∫_σ ||∇L(x,y,σ)||^2 dσ
```

**Multi-Scale Weighting**: Combine edge information across scales:
```
w_j = 1 + ∑_σ α_σ E_σ(patch) corr_j(σ)
```

### 3.4 Coupled Dictionary Learning - Advanced Theory

#### 3.4.1 Manifold Learning Perspective
**Assumption**: LR and HR patches lie on related manifolds:
- M_L: LR patch manifold
- M_H: HR patch manifold
- f: M_L → M_H (unknown mapping)

**Dictionary Learning Goal**: Learn dictionaries that parameterize these manifolds and the mapping between them.

**Sparse Coding**: Finds coordinates on the manifolds:
```
α_L: patch_L → coordinates on M_L
α_H: patch_H → coordinates on M_H
```

**Coupling Constraint**: α_L = α_H = α (same coordinates)

#### 3.4.2 Multi-Task Learning Framework
**Problem Formulation**: Learn multiple related tasks simultaneously:
- Task 1: Represent LR patches sparsely
- Task 2: Represent HR patches sparsely
- Constraint: Shared sparse codes

**Joint Optimization**:
```
min_{D_L,D_H,A} ||Y_L - D_L A||_F^2 + γ||Y_H - D_H A||_F^2 + λ||A||_1
```

**Parameter γ**: Controls relative importance of LR vs HR reconstruction.

**Alternating Minimization**:
1. Fix D_L, D_H: Solve for A (joint sparse coding)
2. Fix A: Update D_L (standard dictionary update)
3. Fix A: Update D_H (least squares solution)

#### 3.4.3 High-Frequency Learning - Spectral Analysis

**Frequency Domain Decomposition**:
```
I_H(ω) = I_L(ω) + I_{HF}(ω)
```
where ω represents spatial frequency.

**Low-Frequency Component**: Can be reconstructed via interpolation
**High-Frequency Component**: Contains edges, textures - needs learning

**Spectral Factorization**:
```
I_{HF}(ω) = H(ω) I_L(ω) + N(ω)
```
where H(ω) is the learned spectral transfer function.

**Dictionary Learning**: Learns H(ω) implicitly through patch relationships.

#### 3.4.4 Generalization Across Scales

**Scale Invariance**: Natural images have statistical self-similarity across scales.

**Multi-Scale Training**: Train on multiple scales simultaneously:
```
min ∑_s ||Y_L^{(s)} - D_L^{(s)} A^{(s)}||_F^2 + ||Y_H^{(s)} - D_H^{(s)} A^{(s)}||_F^2
```

**Shared Structure**: Dictionary atoms at different scales should be related:
```
D_H^{(s)} ≈ scale_transform(D_H^{(s')})
```

### 3.5 Back-Projection and Iterative Refinement

#### 3.5.1 Projection onto Constraint Sets
**Constraint Set**: C = {I : downsample(I) = I_L}

**Projection Operator**: P_C projects any image onto the constraint set:
```
P_C(I) = I + upsample(I_L - downsample(I))
```

**Iterative Projection**: 
```
I^{(t+1)} = P_C(I^{(t)})
```

**Convergence**: Converges to the point in C closest to the initial estimate.

#### 3.5.2 POCS (Projection Onto Convex Sets)
**Multiple Constraints**:
- C_1: Consistency with LR image
- C_2: Smoothness constraints
- C_3: Non-negativity
- C_4: Bounded variation

**POCS Algorithm**:
```
I^{(t+1)} = P_{C_4} P_{C_3} P_{C_2} P_{C_1} I^{(t)}
```

**Convergence**: Converges to intersection of all constraint sets (if non-empty).

#### 3.5.3 Adaptive Step Size Control
**Fixed Step Size**: Can lead to oscillations or slow convergence.

**Adaptive Step Size**:
```
α^{(t)} = α_0 exp(-βt)
```

**Optimal Step Size**: Minimize expected squared error:
```
α* = argmin E[||I^{(t+1)} - I*||^2]
```

**Practical Choice**: α^{(t)} = 0.8 × 0.9^t works well empirically.

---

## 4. Step-by-Step Execution

### 4.1 Training Phase - Mathematical Walkthrough

#### 4.1.1 Data Preprocessing - Signal Processing Perspective

**Image Loading and Normalization**:
```python
I = imread(filename)  # Load as uint8 [0,255]
I = I.astype(float32) / 255.0  # Convert to float [0,1]
```

**Mathematical Justification**: 
- Floating point avoids quantization errors during processing
- [0,1] range prevents numerical overflow in computations
- Preserves relative intensities for gradient computations

**Grayscale Conversion** (if needed):
```python
I_gray = 0.299*R + 0.587*G + 0.114*B  # ITU-R BT.601 standard
```

**Perceptual Basis**: Weights correspond to human luminance sensitivity.

#### 4.1.2 Multi-Resolution Pyramid Construction

**Downsampling with Anti-Aliasing**:
```python
# Gaussian blur to prevent aliasing
I_blurred = gaussian_filter(I, sigma=0.8)
# Downsample by factor s
I_LR = I_blurred[::s, ::s]
```

**Nyquist-Shannon Theorem**: Blurring removes frequencies above π/s to prevent aliasing.

**Alternative: Area-Based Downsampling**:
```python
I_LR = block_reduce(I, (s,s), func=np.mean)
```

**Advantage**: Preserves total energy, reduces noise.

#### 4.1.3 High-Frequency Component Extraction

**Gaussian Blur**:
```python
I_blur = gaussian_filter(I, sigma=1.0)
I_HF = I - I_blur
```

**Frequency Domain Analysis**:
- Gaussian filter: H(ω) = exp(-ω²σ²/2)
- High-pass filter: 1 - H(ω) = 1 - exp(-ω²σ²/2)
- Cutoff frequency: ω_c ≈ √(2 ln(2))/σ

**Alternative: Laplacian of Gaussian**:
```python
I_HF = laplacian_of_gaussian(I, sigma=1.0)
```

**Mathematical Form**: ∇²G(x,y,σ) = -(x²+y²-2σ²)/(2πσ⁶) exp(-(x²+y²)/(2σ²))

#### 4.1.4 Patch Extraction - Sliding Window Analysis

**Sliding Window**:
```python
for i in range(0, H-p+1, step):
    for j in range(0, W-p+1, step):
        patch = I[i:i+p, j:j+p]
```

**Total Patches**: N = ((H-p)/step + 1) × ((W-p)/step + 1)

**Overlap Analysis**:
- Step = 1: Maximum overlap, N ≈ HW
- Step = p: No overlap, N ≈ HW/p²
- Step = p/2: 50% overlap, good balance

**Memory Requirements**: N × p² × sizeof(float32) bytes

#### 4.1.5 Edge-Based Patch Filtering

**Gradient Computation** (Sobel operators):
```python
Gx = [[-1, 0, 1],
      [-2, 0, 2], 
      [-1, 0, 1]]

Gy = [[-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]]

grad_x = convolve2d(patch, Gx)
grad_y = convolve2d(patch, Gy)
magnitude = sqrt(grad_x² + grad_y²)
```

**Edge Strength**:
```python
edge_strength = mean(magnitude)
```

**Statistical Interpretation**: Edge strength measures local variance in gradient space.

**Filtering Criterion**:
```python
if edge_strength > threshold:
    selected_patches.append(patch)
```

**Threshold Selection**: 
- Too low: Include smooth patches (poor for edge learning)
- Too high: Exclude moderate edges (insufficient data)
- Optimal: 10-30% of patches selected

#### 4.1.6 Patch Vectorization and Normalization

**Vectorization**:
```python
patch_vector = patch.reshape(-1)  # p² × 1 vector
```

**Zero-Mean Normalization**:
```python
patch_mean = np.mean(patch_vector)
patch_vector = patch_vector - patch_mean
```

**Mathematical Justification**:
- Removes DC component (illumination invariance)
- Centers data for better dictionary learning
- Reduces dynamic range for numerical stability

**Alternative: Unit Norm**:
```python
patch_vector = patch_vector / (np.linalg.norm(patch_vector) + eps)
```

**Trade-off**: Unit norm vs zero-mean affects dictionary atom interpretation.

### 4.2 Dictionary Learning Phase - K-SVD Implementation

#### 4.2.1 Dictionary Initialization

**Random Initialization**:
```python
# Select random patches as initial atoms
indices = np.random.choice(N, k, replace=False)
D = X[indices].T  # n × k matrix
```

**Normalization**:
```python
D = D / (np.linalg.norm(D, axis=0) + eps)
```

**Alternative: K-means Initialization**:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
D = kmeans.cluster_centers_.T
```

**Advantage**: Better initial clustering, faster convergence.

#### 4.2.2 Sparse Coding Step - OMP Implementation

**Gram Matrix Precomputation**:
```python
G = D.T @ D  # k × k matrix
```

**Batch OMP**:
```python
def batch_omp(D, Y, sparsity):
    n, k = D.shape
    m = Y.shape[1]
    A = np.zeros((k, m))
    
    for i in range(m):
        y = Y[:, i]
        residual = y.copy()
        support = []
        
        for s in range(sparsity):
            # Find most correlated atom
            correlations = np.abs(D.T @ residual)
            best_atom = np.argmax(correlations)
            support.append(best_atom)
            
            # Solve least squares on support
            D_support = D[:, support]
            alpha_support = np.linalg.lstsq(D_support, y, rcond=None)[0]
            
            # Update residual
            residual = y - D_support @ alpha_support
            
            # Store coefficients
            A[support, i] = alpha_support
    
    return A
```

**Computational Complexity**: O(s²nk) per signal, where s is sparsity.

#### 4.2.3 Dictionary Update Step - SVD Analysis

**Atom Update Loop**:
```python
for k in range(num_atoms):
    # Find samples using atom k
    using_k = np.where(np.abs(A[k, :]) > eps)[0]
    
    if len(using_k) == 0:
        # Replace unused atom with random sample
        D[:, k] = random_sample()
        continue
    
    # Compute error without atom k
    D_without_k = D.copy()
    D_without_k[:, k] = 0
    E_k = Y[:, using_k] - D_without_k @ A[:, using_k]
    
    # SVD of error matrix
    U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
    
    # Update atom and coefficients
    D[:, k] = U[:, 0]
    A[k, using_k] = s[0] * Vt[0, :]
```

**SVD Interpretation**:
- U[:, 0]: Direction of maximum variance (new atom)
- s[0]: Amount of variance explained
- Vt[0, :]: Coefficients for samples using this atom

#### 4.2.4 Convergence Monitoring

**Objective Function**:
```python
def compute_objective(D, A, Y):
    reconstruction_error = np.linalg.norm(Y - D @ A, 'fro')**2
    sparsity_penalty = lambda_param * np.sum(np.abs(A))
    return reconstruction_error + sparsity_penalty
```

**Convergence Criteria**:
```python
relative_change = abs(obj_new - obj_old) / obj_old
if relative_change < tolerance:
    break
```

**Typical Values**: tolerance = 1e-4, max_iterations = 20

### 4.3 HR Dictionary Learning - Least Squares Solution

#### 4.3.1 Problem Formulation
Given sparse codes A from LR dictionary learning, find D_H such that:
```
min_{D_H} ||Y_H - D_H A||_F^2
```

**Normal Equations**:
```
D_H A A^T = Y_H A^T
D_H = Y_H A^T (A A^T)^{-1}
```

#### 4.3.2 Numerical Implementation

**Regularized Solution** (for numerical stability):
```python
AAt = A @ A.T + lambda_reg * np.eye(k)
YAt = Y_H @ A.T
D_H = YAt @ np.linalg.inv(AAt)
```

**SVD-Based Solution** (more stable):
```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
# Pseudo-inverse with regularization
s_reg = s / (s**2 + lambda_reg)
A_pinv = Vt.T @ np.diag(s_reg) @ U.T
D_H = Y_H @ A_pinv.T
```

**Condition Number Check**:
```python
cond_num = np.linalg.cond(A @ A.T)
if cond_num > 1e12:
    print("Warning: Ill-conditioned system")
```

#### 4.3.3 Dictionary Normalization

**Column Normalization**:
```python
norms = np.linalg.norm(D_H, axis=0)
D_H = D_H / (norms + eps)
```

**Why Normalize**: 
- Prevents scale ambiguity
- Ensures numerical stability
- Standard convention in dictionary learning

### 4.4 Inference Phase - Reconstruction Pipeline

#### 4.4.1 Test Image Preprocessing

**Consistent Preprocessing**:
```python
# Same as training
I_test = imread(test_file).astype(float32) / 255.0
I_LR = downsample(I_test, scale)  # Create LR input
```

**Patch Extraction**:
```python
patches = []
positions = []
for i in range(0, H_LR - p + 1, step):
    for j in range(0, W_LR - p + 1, step):
        patch = I_LR[i:i+p, j:j+p]
        patches.append(patch.reshape(-1) - np.mean(patch))
        positions.append((i, j))
```

#### 4.4.2 Edge-Aware Weight Computation

**Detailed Implementation**:
```python
def compute_edge_weights(patch, D_L):
    # Reshape patch for gradient computation
    patch_2d = patch.reshape(p, p)
    
    # Compute gradients
    grad_x = np.gradient(patch_2d, axis=1)
    grad_y = np.gradient(patch_2d, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Edge strength
    edge_strength = np.mean(magnitude)
    
    # Texture measures
    patch_std = np.std(patch_2d)
    patch_var = np.var(patch_2d)
    
    # Base weights
    weights = np.ones(D_L.shape[1])
    
    if edge_strength > edge_threshold:
        # Compute correlations
        patch_norm = np.linalg.norm(patch)
        atom_norms = np.linalg.norm(D_L, axis=0)
        correlations = np.abs(D_L.T @ patch) / (patch_norm * atom_norms + eps)
        
        # Combine factors
        edge_factor = min(edge_strength * 3.0, 2.0)
        texture_factor = min(patch_std * 2.0, 1.5)
        
        weights = 1.0 + edge_factor * correlations + texture_factor * correlations
        weights = np.clip(weights, 0.2, 5.0)
    
    return weights
```

#### 4.4.3 Weighted Sparse Coding

**Implementation**:
```python
def weighted_omp(D, y, weights, sparsity):
    if weights is None:
        return standard_omp(D, y, sparsity)
    
    # Clamp weights
    w = np.clip(weights, 0.2, 5.0)
    
    # Scale dictionary
    D_weighted = D * w[None, :]
    
    # Standard OMP on weighted dictionary
    alpha_weighted = standard_omp(D_weighted, y, sparsity)
    
    # Rescale coefficients
    alpha = alpha_weighted * w
    
    return alpha
```

**Mathematical Justification**: This solves the weighted L1 problem:
```
min ||y - Dα||₂² + λ||Wα||₁
```

#### 4.4.4 HR Patch Reconstruction

**Reconstruction**:
```python
def reconstruct_hr_patch(lr_patch, D_L, D_H, use_weights=True):
    # Normalize LR patch
    lr_mean = np.mean(lr_patch)
    lr_normalized = lr_patch - lr_mean
    
    # Compute weights
    if use_weights:
        weights = compute_edge_weights(lr_normalized, D_L)
    else:
        weights = None
    
    # Sparse coding
    alpha = weighted_omp(D_L, lr_normalized, weights, sparsity)
    
    # HR reconstruction
    hr_vector = D_H @ alpha
    hr_patch = hr_vector.reshape(p_hr, p_hr)
    
    # Add back mean (upsampled)
    lr_patch_2d = lr_patch.reshape(p_lr, p_lr)
    upsampled_mean = np.mean(upsample(lr_patch_2d, scale))
    hr_patch = hr_patch + upsampled_mean
    
    return hr_patch
```

#### 4.4.5 Patch Aggregation and Overlap Handling

**Weighted Averaging**:
```python
def aggregate_patches(hr_patches, positions, target_shape, step):
    reconstruction = np.zeros(target_shape)
    weights = np.zeros(target_shape)
    
    for patch, (i, j) in zip(hr_patches, positions):
        # Map LR position to HR position
        hi, hj = i * scale, j * scale
        
        # Add patch to reconstruction
        h_end = min(hi + p_hr, target_shape[0])
        w_end = min(hj + p_hr, target_shape[1])
        
        patch_h = h_end - hi
        patch_w = w_end - hj
        
        reconstruction[hi:h_end, hj:w_end] += patch[:patch_h, :patch_w]
        weights[hi:h_end, hj:w_end] += 1.0
    
    # Normalize by overlap count
    mask = weights > 0
    reconstruction[mask] /= weights[mask]
    
    return reconstruction, mask
```

#### 4.4.6 Back-Projection Refinement

**Iterative Back-Projection**:
```python
def back_projection(hr_estimate, lr_input, scale, iterations=10):
    hr_current = hr_estimate.copy()
    
    for t in range(iterations):
        # Downsample current estimate
        lr_estimate = downsample(hr_current, scale)
        
        # Compute error
        error = lr_input - lr_estimate
        
        # Upsample error
        error_upsampled = upsample(error, scale)
        
        # Adaptive step size
        step_size = 0.8 * (0.9 ** t)
        
        # Update estimate
        hr_current = hr_current + step_size * error_upsampled
        
        # Clamp to valid range
        hr_current = np.clip(hr_current, 0.0, 1.0)
    
    return hr_current
```

**Convergence Analysis**: The sequence {hr_current} converges to the projection of the initial estimate onto the constraint set {I : downsample(I) = lr_input}.

This completes the deep mathematical analysis of your edge-aware dictionary learning project. The documentation now provides comprehensive theoretical foundations and detailed algorithmic explanations at the mathematical level you requested.

# Global configuration - edit paths and hyperparams here
import os

# Auto-detect project root and set paths relative to it
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths (automatically work locally, on Colab, and Kaggle)
HR_TRAIN_DIR = os.path.join(PROJECT_ROOT, "train_512")       # put HR training images here (BSD100/Urban100)
HR_TEST_DIR  = os.path.join(PROJECT_ROOT, "test_data200")        # put HR test images here
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")              # where results and dicts will be saved

# Scales and sizes
SCALES = [2, 3, 4]                   # scales to train/evaluate
LR_PATCH_SIZE = 7                    # Increased for better context
HR_PAD = 4                           # Increased padding for better reconstruction
UPSCALE_PAD = 2                      # unused for now, kept for compatibility

# Dictionary / K-SVD params
DICT_ATOMS = 256                    # Reduced to avoid linear dependence
SPARSITY = 4                         # Reduced for smaller dictionary
KSVD_ITERS = 8                    # Reduced iterations

# Edge filtering
EDGE_THRESHOLD = 0.3                 # Lowered to include more edge patches

# Inference / aggregation
PATCH_STEP = 1                       # Smaller step for better reconstruction
BACKPROJECTION_ITERS = 10           # More iterations for refinement

# Patch budget (None or 0 = no cap)
MAX_PATCHES = 600000                     # Use all available patches from all images

# Misc
RANDOM_SEED = 42
N_JOBS = -1                           # for sklearn OMP etc, -1 uses all cores


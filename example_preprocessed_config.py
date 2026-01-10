# Example Configuration for Pre-Processed Data

# ====================================================================
# COPY THESE LINES TO YOUR NOTEBOOK (Section 5)
# ====================================================================

# %% [code]
# ============================================
# 🔧 PRE-PROCESSED DATA CONFIGURATION
# ============================================

# Set to True to use pre-processed data
USE_PREPROCESSED = True

# Path to your uploaded .npz files
# Replace with your Kaggle dataset path after uploading
PREPROCESSED_INPUT = "/kaggle/input/av2-processed-train/processed_data/"

# ============================================
# 🔀 OPTIONAL: MERGE MULTIPLE BATCHES
# ============================================

# If you have multiple batches from incremental processing:
MERGE_BATCHES = False  # Set to True if you want to merge

# Add your batch directories here
BATCH_DIRS = [
    "/kaggle/input/av2-batch1/processed_data/",
    "/kaggle/input/av2-batch2/processed_data/",
    "/kaggle/input/av2-batch3/processed_data/",
    # Add more as needed...
]

# Output directory for merged data
OUTPUT_MERGED = "/kaggle/working/merged_processed_data"

# ====================================================================
# AUTOMATIC CONFIGURATION (don't change)
# ====================================================================

if MERGE_BATCHES:
    print("🔀 Merging batches...")
    import subprocess
    
    # Run merge script
    cmd = [
        "python", "merge_processed_batches.py",
        "--input_dirs"
    ] + BATCH_DIRS + [
        "--output", OUTPUT_MERGED
    ]
    
    subprocess.run(cmd, check=True)
    
    # Use merged data
    PREPROCESSED_INPUT = OUTPUT_MERGED
    print(f"✅ Merged data ready: {PREPROCESSED_INPUT}")

# ====================================================================
# EXAMPLE CONFIGURATIONS
# ====================================================================

# Example 1: Single upload (most common)
"""
USE_PREPROCESSED = True
PREPROCESSED_INPUT = "/kaggle/input/av2-train-processed-v1/processed_data/"
MERGE_BATCHES = False
"""

# Example 2: Merge 3 batches
"""
USE_PREPROCESSED = True
MERGE_BATCHES = True
BATCH_DIRS = [
    "/kaggle/input/av2-batch1/processed_data/",
    "/kaggle/input/av2-batch2/processed_data/",
    "/kaggle/input/av2-batch3/processed_data/",
]
"""

# Example 3: Process fresh data (no pre-processed)
"""
USE_PREPROCESSED = False
PREPROCESSED_INPUT = None
MERGE_BATCHES = False
"""

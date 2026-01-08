# %% [markdown]
# # TrafficGamer (QCNet) + Argoverse 2 - Kaggle Notebook
# 
# Notebook này load QCNet pretrained và evaluate trên Argoverse 2 dataset.
# 
# **Repo:** https://github.com/PhamPhuHoa-23/TrafficGamer (Modified QCNet cho Kaggle)
# 
# **Dataset cần add trong Kaggle:**
# - `nek-chua` hoặc dataset chứa Argoverse 2 train/val data
# - `qcnetckptargoverse` - checkpoint QCNet AV2

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Install các packages cần thiết
!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning==2.0.0
!pip install -q torch-geometric
!pip install -q av2  # Argoverse 2 API
!pip install -q pyarrow  # Để đọc parquet files

# %%
# Install PyG dependencies (quan trọng cho QCNet)
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Lấy CUDA version để install đúng packages
cuda_version = torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
print(f"CUDA version: {cuda_version}")

# %%
# Install torch-scatter, torch-sparse, torch-cluster
!pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_version[:3]}.html

# %% [markdown]
# ## 2. Import Libraries

# %%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# %% [markdown]
# ## 3. Clone TrafficGamer Repository (Modified QCNet)

# %%
import os

# Clone TrafficGamer repo (QCNet đã được sửa cho Argoverse 2 trên Kaggle)
if not os.path.exists('TrafficGamer'):
    !git clone https://github.com/PhamPhuHoa-23/TrafficGamer.git
    print("✅ TrafficGamer cloned")
else:
    print("✅ TrafficGamer already exists")

# Change directory to TrafficGamer
os.chdir('TrafficGamer')
print(f"📁 Current directory: {os.getcwd()}")

# %%
# Thêm TrafficGamer vào path
import sys
sys.path.insert(0, os.getcwd())

# %% [markdown]
# ## 4. Setup Data Paths
# 
# **Thay đổi paths này theo dataset của bạn trên Kaggle**

# %%
# ============================================
# 🔧 THAY ĐỔI PATHS Ở ĐÂY THEO KAGGLE DATASET
# ============================================

# Path đến Argoverse 2 data
AV2_ROOT = "/kaggle/input/nek-chua"  # Thay đổi theo dataset của bạn

# Path đến checkpoint
CKPT_PATH = "/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt"

# Hoặc download từ HuggingFace
# !wget -q https://huggingface.co/ZikangZhou/QCNet/resolve/main/QCNet_AV2.ckpt -O qcnet_av2.ckpt
# CKPT_PATH = "qcnet_av2.ckpt"

print(f"📂 AV2 Root: {AV2_ROOT}")
print(f"📦 Checkpoint: {CKPT_PATH}")

# %% [markdown]
# ## 5. Load và Cache Scenarios
# 
# **Lưu ý:** Glob rất lâu (~vài phút), nên save vào pkl để dùng lại.
# - Nếu đã có file pkl glob sẵn → load từ pkl (nhanh)
# - Nếu chưa có → glob và save pkl

# %%
import pickle

# ============================================
# 🔧 PATHS CHO CACHED SCENARIOS
# ============================================
# Path đến file pkl đã glob sẵn (nếu có upload lên Kaggle dataset)
CACHED_SCENARIOS_INPUT = "/kaggle/input/argoverse-glob/av2_scenarios.pkl"

# Path để save pkl mới (nếu cần glob)
CACHED_SCENARIOS_OUTPUT = "av2_scenarios.pkl"

# %%
def load_or_glob_scenarios(cached_input, cached_output, data_root, split='train'):
    """
    Load scenarios từ cached pkl hoặc glob mới nếu chưa có.
    
    Priority:
    1. Load từ cached input (uploaded dataset)
    2. Load từ cached output (đã glob trước đó trong session)
    3. Glob mới và save
    """
    # 1. Try load từ input dataset (đã upload sẵn)
    if Path(cached_input).exists():
        print(f"📦 Loading cached scenarios from: {cached_input}")
        with open(cached_input, 'rb') as f:
            scenarios = pickle.load(f)
        print(f"✅ Loaded {len(scenarios)} scenarios (cached)")
        return scenarios
    
    # 2. Try load từ output (đã glob trong session này)
    if Path(cached_output).exists():
        print(f"📦 Loading scenarios from: {cached_output}")
        with open(cached_output, 'rb') as f:
            scenarios = pickle.load(f)
        print(f"✅ Loaded {len(scenarios)} scenarios")
        return scenarios
    
    # 3. Glob mới (chậm)
    print("🔍 Globbing scenarios... (this may take a few minutes)")
    root = Path(data_root)
    
    # Thử nhiều patterns (Kaggle thường có nested structure)
    patterns = [
        f"{split}/{split}/*/scenario_*.parquet",   # Kaggle: train/train/xxx/ ← Try this first!
        f"{split}/*/scenario_*.parquet",            # Standard: train/xxx/
        f"**/scenario_*.parquet",                   # Recursive search (fallback)
    ]
    
    scenarios = []
    for pattern in patterns:
        print(f"  Trying pattern: {pattern}")
        found = list(root.glob(pattern))
        if found:
            scenarios = found
            print(f"  ✅ Found {len(scenarios)} scenarios")
            break
    
    if not scenarios:
        print("⚠️ No scenarios found! Listing directory structure:")
        for item in list(root.iterdir())[:5]:
            print(f"  - {item}")
            if item.is_dir():
                for subitem in list(item.iterdir())[:3]:
                    print(f"      - {subitem}")
        return []
    
    # Save để dùng lại
    print(f"💾 Saving to {cached_output}...")
    with open(cached_output, 'wb') as f:
        pickle.dump(scenarios, f)
    print(f"✅ Saved {len(scenarios)} scenarios")
    
    return scenarios

# %%
# Load scenarios
scenarios = load_or_glob_scenarios(
    cached_input=CACHED_SCENARIOS_INPUT,
    cached_output=CACHED_SCENARIOS_OUTPUT,
    data_root=AV2_ROOT,
    split='train'
)

print(f"\n📊 Total scenarios: {len(scenarios)}")
if scenarios:
    print(f"📄 Example: {scenarios[0]}")

# %% [markdown]
# ## 6. Hiểu Argoverse 2 Data Format
# 
# Argoverse 2 parquet file chứa các columns:
# - `track_id`: ID của mỗi agent
# - `object_type`: vehicle, pedestrian, cyclist, etc.
# - `object_category`: FOCAL_TRACK, SCORED_TRACK, UNSCORED_TRACK
# - `timestep`: 0-109 (110 timesteps @ 10Hz = 11 seconds)
# - `position_x`, `position_y`: Tọa độ
# - `heading`: Hướng (radians)
# - `velocity_x`, `velocity_y`: Vận tốc

# %%
def load_scenario(parquet_path):
    """Load một scenario từ parquet file."""
    df = pd.read_parquet(parquet_path)
    return df

# %%
# Load và xem một scenario mẫu
if scenarios:
    sample_df = load_scenario(scenarios[0])
    print("📋 Columns:")
    print(sample_df.columns.tolist())
    print("\n📊 Shape:", sample_df.shape)
    print("\n🔍 Sample data:")
    print(sample_df.head(10))
    print("\n📈 Object categories:")
    print(sample_df['object_category'].value_counts())
    print("\n🚗 Object types:")
    print(sample_df['object_type'].value_counts())

# %% [markdown]
# ## 7. Process Argoverse 2 Data cho QCNet

# %%
def process_av2_scenario(df):
    """
    Extract features từ Argoverse 2 parquet cho QCNet inference.
    
    Argoverse 2 timeline:
    - Timesteps 0-49: History (5 seconds @ 10Hz)
    - Timesteps 50-109: Future to predict (6 seconds @ 10Hz)
    
    Returns:
        agent_hist: (50, 5) - [x, y, vx, vy, heading]
        gt_future: (60, 2) - [x, y]
        all_agents_hist: Optional - history of all agents
    """
    # ===== 1. Tìm focal agent =====
    # Convert object_category to string to handle different types
    obj_cat = df['object_category'].astype(str).str.upper()
    
    # Try different variations
    focal_mask = obj_cat == 'FOCAL_TRACK'
    if not focal_mask.any():
        focal_mask = obj_cat == 'SCORED_TRACK'
    if not focal_mask.any():
        focal_mask = obj_cat == 'TRACK_FRAGMENT'
    if not focal_mask.any():
        # Fallback: dùng track có nhiều timesteps nhất
        track_counts = df.groupby('track_id').size()
        main_track = track_counts.idxmax()
        focal_mask = df['track_id'] == main_track
        print(f"⚠️ No focal track found, using main track: {main_track}")
    
    focal_df = df[focal_mask].sort_values('timestep')
    
    if len(focal_df) == 0:
        raise ValueError("No focal agent found in scenario")
    
    # ===== 2. Extract history (timesteps 0-49) =====
    hist_df = focal_df[focal_df['timestep'] < 50].copy()
    
    # Xử lý missing timesteps
    required_cols = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading']
    
    # Check và handle missing columns
    for col in required_cols:
        if col not in hist_df.columns:
            if col == 'velocity_x':
                hist_df['velocity_x'] = hist_df['position_x'].diff() * 10  # 10Hz
            elif col == 'velocity_y':
                hist_df['velocity_y'] = hist_df['position_y'].diff() * 10
            elif col == 'heading':
                hist_df['heading'] = np.arctan2(
                    hist_df['velocity_y'].fillna(0),
                    hist_df['velocity_x'].fillna(0)
                )
    
    agent_hist = hist_df[required_cols].fillna(0).values
    
    # Pad nếu thiếu timesteps (phải có đủ 50 frames)
    if len(agent_hist) < 50:
        pad = np.zeros((50 - len(agent_hist), 5))
        agent_hist = np.vstack([pad, agent_hist])
    elif len(agent_hist) > 50:
        agent_hist = agent_hist[-50:]  # Lấy 50 frames cuối
    
    # ===== 3. Extract ground truth future (timesteps 50-109) =====
    future_df = focal_df[focal_df['timestep'] >= 50].copy()
    
    if len(future_df) == 0:
        gt_future = np.zeros((60, 2))
    else:
        gt_future = future_df[['position_x', 'position_y']].values
        
        # Pad nếu thiếu
        if len(gt_future) < 60:
            last_pos = gt_future[-1] if len(gt_future) > 0 else np.zeros(2)
            pad = np.tile(last_pos, (60 - len(gt_future), 1))
            gt_future = np.vstack([gt_future, pad])
        elif len(gt_future) > 60:
            gt_future = gt_future[:60]
    
    return agent_hist.astype(np.float32), gt_future.astype(np.float32)

# %%
# Test processing
if scenarios:
    sample_df = load_scenario(scenarios[0])
    agent_hist, gt_future = process_av2_scenario(sample_df)
    print(f"✅ Agent history shape: {agent_hist.shape}")  # Expected: (50, 5)
    print(f"✅ Ground truth future shape: {gt_future.shape}")  # Expected: (60, 2)
    print(f"\n📍 First history point: {agent_hist[0]}")
    print(f"📍 Last history point: {agent_hist[-1]}")
    print(f"📍 First future point: {gt_future[0]}")

# %% [markdown]
# ## 8. Load QCNet Model

# %%
# Import QCNet
try:
    from predictors import QCNet
    print("✅ QCNet imported successfully")
except ImportError as e:
    print(f"❌ Error importing QCNet: {e}")
    print("Trying alternative import...")
    from predictors.qcnet import QCNet

# %%
# Load model từ checkpoint
if os.path.exists(CKPT_PATH):
    model = QCNet.load_from_checkpoint(CKPT_PATH)
    model.eval()
    model = model.to(device)
    print(f"✅ Model loaded from {CKPT_PATH}")
else:
    print(f"❌ Checkpoint not found: {CKPT_PATH}")
    print("Downloading from HuggingFace...")
    !wget -q https://huggingface.co/ZikangZhou/QCNet/resolve/main/QCNet_AV2.ckpt -O qcnet_av2.ckpt
    model = QCNet.load_from_checkpoint('qcnet_av2.ckpt')
    model.eval()
    model = model.to(device)
    print("✅ Model loaded from HuggingFace")

# %% [markdown]
# ## 9. Evaluation Metrics

# %%
from scipy.stats import gaussian_kde

def compute_ade(pred_trajs, gt_traj):
    """
    Average Displacement Error.
    pred_trajs: (num_modes, num_steps, 2)
    gt_traj: (num_steps, 2)
    """
    # Broadcast gt to match pred shape
    errors = np.linalg.norm(pred_trajs - gt_traj[None, :, :], axis=-1)  # (num_modes, num_steps)
    ade_per_mode = errors.mean(axis=1)  # (num_modes,)
    return ade_per_mode.min()  # minADE

def compute_fde(pred_trajs, gt_traj):
    """
    Final Displacement Error.
    """
    final_errors = np.linalg.norm(pred_trajs[:, -1, :] - gt_traj[-1, :], axis=-1)  # (num_modes,)
    return final_errors.min()  # minFDE

def compute_mr(pred_trajs, gt_traj, threshold=2.0):
    """
    Miss Rate - tỷ lệ predictions có FDE > threshold.
    """
    final_errors = np.linalg.norm(pred_trajs[:, -1, :] - gt_traj[-1, :], axis=-1)
    return float(final_errors.min() > threshold)

def compute_nll_kde(pred_trajs, gt_traj, bandwidth=0.1):
    """
    Negative Log-Likelihood via Kernel Density Estimation.
    Đo uncertainty của predictions.
    """
    num_modes, num_steps, _ = pred_trajs.shape
    num_gt_steps = len(gt_traj)
    
    nll_total = 0.0
    valid_steps = 0
    
    for t in range(min(num_steps, num_gt_steps)):
        pred_points = pred_trajs[:, t, :].T  # (2, num_modes)
        gt_point = gt_traj[t]
        
        # Skip nếu predictions quá gần nhau (KDE sẽ fail)
        if np.std(pred_points) < 1e-6:
            continue
        
        try:
            kde = gaussian_kde(pred_points, bw_method=bandwidth)
            likelihood = kde(gt_point)[0]
            likelihood = max(likelihood, 1e-10)  # Avoid log(0)
            nll_total += -np.log2(likelihood)
            valid_steps += 1
        except:
            continue
    
    return nll_total / max(valid_steps, 1)

# %% [markdown]
# ## 10. Run Evaluation - Verify & Apply Patches
# 
# **📝 Note:** Nếu dùng TrafficGamer repo (đã patched), section này chỉ verify.
# Nếu dùng original QCNet, section này sẽ apply các patches cần thiết:
# 1. `TargetBuilder`: Thêm `forward()` method
# 2. `ArgoverseV2Dataset`: Disable auto-download cho Kaggle read-only filesystem

# %%
# ===== COMPREHENSIVE PATCHES =====
import oVERIFY & APPLY PATCHES =====
import os
import re
from pathlib import Path

print("🔧 Verifying and applying patches (if needed)
# ===== PATCH 1: TargetBuilder - Add forward() method =====
target_builder_file = Path(os.getcwd()) / 'transforms' / 'target_builder.py'

if target_builder_file.exists():
    print(f"1️⃣ Patching TargetBuilder...")
    content = target_builder_file.read_text()
    
    if 'def forward(self' not in content:
        # Add forward method before __call__
        patch = '''    def forward(self, data):
        """Pass-through implementation for abstract method."""
        return data

    '''
        if 'def __call__(self' in content:
            content = content.replace('def __call__(self', patch + 'def __call__(self')
            target_builder_file.write_text(content)
            print("   ✅ Added forward() method to TargetBuilder")
        else:
            print("   ⚠️ Could not find __call__ method")
    else:
        print("   ✅ TargetBuilder already patched")
else:
    print(f"   ❌ Not found: {target_builder_file}")

# ===== PATCH 2: ArgoverseV2Dataset - Disable auto-download =====
dataset_file = Path(os.getcwd()) / 'datasets' / 'argoverse_v2_dataset.py'

if dataset_file.exists():
    print(f"\n2️⃣ Patching ArgoverseV2Dataset...")
    content = dataset_file.read_text()
    
    if '# KAGGLE_PATCHED' not in content:
        # Method 1: Override _download() method
        if 'def _download(self):' in content:
            old_download = '''    def _download(self):
        if files_exist(self.processed_paths):
            return
        self.download()'''
            
            new_download = '''    def _download(self):  # KAGGLE_PATCHED
        # Skip auto-download on Kaggle (read-only input)
        if not files_exist(self.processed_paths):
            print(f"⚠️ Processed files not found: {self.processed_paths}")
            print(f"⚠️ This is normal on Kaggle - using raw parquet files directly")
        return  # Skip download'''
            
            content = content.replace(old_download, new_download)
        
        # Method 2: Override download() method as well
        if 'def download(self):' in content:
            # Find the download method and replace it
            pattern = r'(    def download\(self\):.*?)(?=\n    def |\nclass |\Z)'
            replacement = '''    def download(self):  # KAGGLE_PATCHED
        """Disabled for Kaggle - data should be in input directory"""
        print("⚠️ Skipping download (Kaggle read-only filesystem)")
        return'''
            
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        dataset_file.write_text(content)
        print("   ✅ Disabled auto-download in ArgoverseV2Dataset")
    else:
        print("   ✅ ArgoverseV2Dataset already patched")
else:
    print(f"   ❌ Not found: {dataset_file}")

# ===== PATCH 3: Use raw data directly (skip processed files requirement) =====
print(f"\n3️⃣ Checking data structure...")

# Point val.py to the actual raw data location
input_data_path = Path(AV2_ROOT)

# Find actual scenario files
possible_splits = ['train', 'val', 'test']
found_data = False

for split in possible_splits:
    # Try different nesting levels
    for nested_path in [
        input_data_path / split / split,  # Kaggle: train/train/
        input_data_path / split,           # Standard: train/
        input_data_path,                   # Root
    ]:
        if nested_path.exists():
            scenarios = list(nested_path.glob("*/scenario_*.parquet"))
            if scenarios:
                print(f"   ✅ Found {len(scenarios)} scenarios in: {nested_path}")
                print(f"   📍 Split: {split}")
                found_data = True
                
                # Set this as the root for validation
                ACTUAL_DATA_ROOT = str(nested_path.parent)
                ACTUAL_SPLIT = split
                break
    if found_data:
        break

if not found_data:
    print("   ⚠️ No scenario parquet files found!")
    print("   Listing directory structure:")
    for item in list(input_data_path.iterdir())[:5]:
        print(f"      - {item}")
        if item.is_dir():
            for sub in list(item.iterdir())[:3]:
                print(f"         - {sub}")
    ACTUAL_DATA_ROOT = AV2_ROOT
    ACTUAL_SPLIT = "train"

print(f"\n📊 Final configuration:")
print(f"   Data root: {ACTUAL_DATA_ROOT}")
print(f"   Split: {ACTUAL_SPLIT}")

# %%
# Run validation with actual data location
print(f"\n{'='*60}")
print(f"🚀 Running QCNet Validation")
print(f"{'='*60}\n"TrafficGamer/QCNet Validation")
print(f"{'='*60}\n")

!python val.py --model QCNet --root {ACTUAL_DATA_ROOT} --ckpt_path {CKPT_PATH}

print(f"\n{'='*60}")
print("Expected metrics:")
print("  minFDE(K=6): ~1.25")
print("  minADE(K=6): ~0.72")
print("  MR(K=6): ~0.16")
print(f"
# Expected output:
# minFDE(K=6): ~1.25
# minADE(K=6): ~0.72  
# MR(K=6): ~0.16

# %% [markdown]
# ## 11. Alternative Evaluation Methods
# 
# **Note:** DataModule approach có thể gặp lỗi `TargetBuilder` abstract class.  
# Nếu Section 10 không hoạt động, dùng manual evaluation ở Section 12 bên dưới.

# %%
# ⚠️ DataModule approach (có thể gặp lỗi TargetBuilder)
# Uncomment nếu muốn thử:

# from datamodules import ArgoverseV2DataModule

# datamodule = ArgoverseV2DataModule(
#     root=AV2_ROOT,
#     train_batch_size=1,
#     val_batch_size=1,
#     test_batch_size=1,
#     num_workers=0,
# )

# datamodule.setup('validate')
# val_loader = datamodule.val_dataloader()
# print(f"✅ Validation loader ready: {len(val_loader)} batches")

# %% [markdown]
# ## 12. Manual Evaluation (Backup Method)
# 
# Dùng cách này nếu val.py script không hoạt động.  
# **⚠️ Warning:** Manual evaluation không bao gồm map data → kết quả không chính xác!

# %%
def run_manual_evaluation(model, scenarios, device, num_samples=20):
    """
    Test metrics với dummy predictions.
    
    ⚠️ Chỉ để test metrics functions, KHÔNG phải official evaluation!
    QCNet cần full data (agent + map + neighbors) để predict chính xác.
    """
    results = {
        'minADE': [],
        'minFDE': [],
        'MR': [],
    }
    
    print(f"🔄 Testing metrics với {num_samples} scenarios...")
    for scenario_path in tqdm(scenarios[:num_samples]):
        try:
            df = load_scenario(scenario_path)
            agent_hist, gt_future = process_av2_scenario(df)
            
            # Dummy predictions xung quanh GT
            pred_np = np.zeros((6, 60, 2))
            for mode_idx in range(6):
                noise = np.random.randn(60, 2) * 1.0
                offset = np.random.randn(2) * 3.0
                pred_np[mode_idx] = gt_future + noise + offset
            
            results['minADE'].append(compute_ade(pred_np, gt_future))
            results['minFDE'].append(compute_fde(pred_np, gt_future))
            results['MR'].append(compute_mr(pred_np, gt_future))
            
        except Exception as e:
            continue
    
    return results

# %%
# Run manual evaluation (chỉ để test)
if scenarios:
    print("\\n⚠️ Running MANUAL evaluation (dummy predictions - testing only)")
    print("⚠️ For official metrics, use Section 10: val.py script\\n")
    
    manual_results = run_manual_evaluation(
        model=model,
        scenarios=scenarios,
        device=device,
        num_samples=20
    )
    
    print("\\n" + "="*50)
    print("📊 MANUAL TEST RESULTS (Not Real Predictions)")
    print("="*50)
    print(f"minADE: {np.mean(manual_results['minADE']):.3f} ± {np.std(manual_results['minADE']):.3f}")
    print(f"minFDE: {np.mean(manual_results['minFDE']):.3f} ± {np.std(manual_results['minFDE']):.3f}")
    print(f"MR:     {np.mean(manual_results['MR']):.3f} ± {np.std(manual_results['MR']):.3f}")
    print("="*50)
    print("\\n⚠️ Use val.py script for real evaluation!")

# %% [markdown]
# ## 13. Visualization

# %%
import matplotlib.pyplot as plt

def visualize_prediction(agent_hist, gt_future, pred_trajs, pred_probs=None):
    """Visualize history, ground truth, và predictions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # History
    hist_xy = agent_hist[:, :2]
    ax.plot(hist_xy[:, 0], hist_xy[:, 1], 'b-', linewidth=2, label='History')
    ax.scatter(hist_xy[-1, 0], hist_xy[-1, 1], c='blue', s=100, zorder=5, marker='o')
    
    # Ground truth future
    ax.plot(gt_future[:, 0], gt_future[:, 1], 'g-', linewidth=2, label='Ground Truth')
    ax.scatter(gt_future[-1, 0], gt_future[-1, 1], c='green', s=100, zorder=5, marker='*')
    
    # Predictions
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(pred_trajs)))
    for i, pred in enumerate(pred_trajs):
        prob_str = f" ({pred_probs[i]:.2f})" if pred_probs is not None else ""
        ax.plot(pred[:, 0], pred[:, 1], '-', color=colors[i], 
                linewidth=1.5, alpha=0.7, label=f'Pred {i+1}{prob_str}')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best')
    ax.set_title('Motion Prediction')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# %%
# Visualize một sample với real predictions (nếu có thể)
if scenarios:
    sample_df = load_scenario(scenarios[0])
    agent_hist, gt_future = process_av2_scenario(sample_df)
    
    # Try to get real predictions từ model
    try:
        # ⚠️ Lưu ý: QCNet cần full batch data (map, neighbors, etc.)
        # Đây là simplified version nên có thể không chính xác
        
        # Prepare input (simplified - thiếu map data)
        from datasets import ArgoverseV2Dataset
        
        # Load scenario với proper format
        scenario_id = scenarios[0].parent.name
        
        # Fallback: Dùng dummy predictions để visualize
        print("⚠️ Using dummy predictions for visualization")
        print("⚠️ For real predictions, run proper inference with full data")
        
        pred_trajs = np.zeros((6, 60, 2))
        for i in range(6):
            noise = np.random.randn(60, 2) * 0.5
            offset = np.random.randn(2) * 2.0
            pred_trajs[i] = gt_future + noise + offset
        
    except Exception as e:
        print(f"⚠️ Cannot run model inference: {e}")
        print("⚠️ Using dummy predictions for visualization demo")
        
        pred_trajs = np.zeros((6, 60, 2))
        for i in range(6):
            noise = np.random.randn(60, 2) * 0.5
            offset = np.random.randn(2) * 2.0
            pred_trajs[i] = gt_future + noise + offset
    
    visualize_prediction(agent_hist, gt_future, pred_trajs)

# %% [markdown]
# ## 14. Save Results

# %%
# Save evaluation results
import json

if 'manual_results' in locals():
    save_results = {
        'method': 'manual_test',
        'warning': 'Dummy predictions for testing only - NOT real model output',
        'minADE': float(np.mean(manual_results.get('minADE', [0]))),
        'minFDE': float(np.mean(manual_results.get('minFDE', [0]))),
        'MR': float(np.mean(manual_results.get('MR', [0]))),
        'num_samples': len(manual_results.get('minADE', [])),
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("✅ Results saved to evaluation_results.json")
    print(json.dumps(save_results, indent=2))
else:
    print("⚠️ No results to save. Run evaluation first!")
    print("   Use Section 10 (val.py) for official evaluation.")

# %% [markdown]
# ## 📝 Notes
# 
# ### Argoverse 2 Data Format:
# - Parquet files chứa trajectory data
# - 110 timesteps (50 history + 60 future) @ 10Hz
# - Columns: track_id, object_type, object_category, timestep, position_x, position_y, heading, velocity_x, velocity_y
# 
# ### QCNet Expected Input:
# - QCNet cần đầy đủ features: agent history, map data, neighbor agents
# - Dùng `ArgoverseV2DataModule` để load data đúng format
# - Hoặc xem `datasets/argoverse_v2_dataset.py` để biết cách process
# 
# ### Troubleshooting:
# 1. **Checkpoint error**: Download lại từ HuggingFace
# 2. **Data not found**: Check path structure, dùng `glob` để tìm files
# 3. **CUDA OOM**: Giảm batch size hoặc dùng CPU
# 4. **Import error**: Check PyG dependencies (torch-scatter, etc.)

# %% [markdown]
# ## Done! 🎉

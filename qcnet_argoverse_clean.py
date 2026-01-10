# %% [markdown]
# # TrafficGamer (QCNet) + Argoverse 2 - Kaggle Notebook (Clean Version)
#
# Load QCNet pretrained và train/evaluate trên pre-processed Argoverse 2 data.
#
# **Repo:** https://github.com/PhamPhuHoa-23/TrafficGamer
#
# **Required Datasets:**
# - Pre-processed train data (.npz files)
# - Pre-processed val data (.npz files)
# - QCNet checkpoint

# %% [markdown]
# ## 1. Install Dependencies

# %% [code]
import torch
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import os
import sys
import warnings

!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning==2.0.0
!pip install -q torch-geometric
!pip install -q av2
!pip install -q pyarrow

# %% [code]
# Install PyG dependencies
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

cuda_version = torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
!pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+{cuda_version[:3]}.html

# %% [markdown]
# ## 2. Setup Environment

# %% [code]
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

# Clone TrafficGamer repo
if not os.path.exists('TrafficGamer'):
    !git clone https://github.com/PhamPhuHoa-23/TrafficGamer.git
    
os.chdir('TrafficGamer')
sys.path.insert(0, os.getcwd())
print(f"📁 Working dir: {os.getcwd()}")

# Install requirements
!pip install -r requirements.txt
!pip install neptune

# %% [markdown]
# ## 3. Configure Paths
#
# **⚠️ IMPORTANT: Update these paths with your Kaggle datasets**

# %% [code]
# ============================================
# 🔧 CONFIGURE YOUR PATHS HERE
# ============================================

# Pre-processed data (upload your .npz files to Kaggle datasets)
TRAIN_PROCESSED_NPZ = "/kaggle/input/av2-processed-train/processed_data/"
VAL_PROCESSED_NPZ = "/kaggle/input/av2-processed-val/processed_data/"

# Checkpoint
CKPT_PATH = "/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt"

# Training config
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
NUM_WORKERS = 2
MAX_EPOCHS = 10

print(f"📂 Train data: {TRAIN_PROCESSED_NPZ}")
print(f"📂 Val data: {VAL_PROCESSED_NPZ}")
print(f"📦 Checkpoint: {CKPT_PATH}")

# Verify paths
assert Path(TRAIN_PROCESSED_NPZ).exists(), f"Train data not found: {TRAIN_PROCESSED_NPZ}"
assert Path(VAL_PROCESSED_NPZ).exists(), f"Val data not found: {VAL_PROCESSED_NPZ}"
assert Path(CKPT_PATH).exists(), f"Checkpoint not found: {CKPT_PATH}"

print("✅ All paths verified!")

# %% [markdown]
# ## 4. Load Model

# %% [code]
from predictors import QCNet

model = QCNet.load_from_checkpoint(CKPT_PATH)
model.eval()
model = model.to(device)
print(f"✅ Model loaded from {CKPT_PATH}")

# %% [markdown]
# ## 5. Setup DataModule với Pre-Processed Data

# %% [code]
from datamodules import ArgoverseV2DataModule

# Create datamodule pointing to pre-processed .npz directories
datamodule = ArgoverseV2DataModule(
    root='/',  # Dummy root
    train_batch_size=TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    test_batch_size=VAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    
    # Point to pre-processed directories
    train_raw_dir=None,
    train_processed_dir=TRAIN_PROCESSED_NPZ,
    val_raw_dir=None,
    val_processed_dir=VAL_PROCESSED_NPZ,
    
    # AV2 config
    dim=3,
    num_historical_steps=50,
    num_future_steps=60,
)

# Setup datasets
datamodule.setup('fit')
print(f"✅ Train dataset: {len(datamodule.train_dataloader())} batches")
print(f"✅ Val dataset: {len(datamodule.val_dataloader())} batches")

# %% [markdown]
# ## 6. Training/Fine-tuning

# %% [code]
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='qcnet-{epoch:02d}-{val_loss:.3f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Logger
logger = TensorBoardLogger('lightning_logs', name='qcnet_av2')

# Trainer
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[checkpoint_callback, lr_monitor],
    logger=logger,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
)

# Train
print("🚀 Starting training...")
trainer.fit(model, datamodule)

print("✅ Training complete!")
print(f"📂 Best checkpoint: {checkpoint_callback.best_model_path}")

# %% [markdown]
# ## 7. Evaluation

# %% [code]
# Load best checkpoint
best_model = QCNet.load_from_checkpoint(checkpoint_callback.best_model_path)
best_model.eval()
best_model = best_model.to(device)

# Evaluate
print("🔍 Evaluating on validation set...")
results = trainer.validate(best_model, datamodule)

print("\n📊 Validation Results:")
for key, value in results[0].items():
    print(f"  {key}: {value:.4f}")

# %% [markdown]
# ## 8. Save Results

# %% [code]
import json

results_dict = {
    'best_checkpoint': str(checkpoint_callback.best_model_path),
    'validation_metrics': results[0],
    'config': {
        'train_data': TRAIN_PROCESSED_NPZ,
        'val_data': VAL_PROCESSED_NPZ,
        'batch_size': TRAIN_BATCH_SIZE,
        'epochs': MAX_EPOCHS,
    }
}

with open('training_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("✅ Results saved to training_results.json")

# %% [markdown]
# ## Done! 🎉
#
# ### Next Steps:
# 1. Download checkpoints from `/kaggle/working/checkpoints/`
# 2. Download training logs from `/kaggle/working/lightning_logs/`
# 3. Use best checkpoint for inference
#
# ### Notes:
# - Pre-processed .npz files are loaded directly (no processing!)
# - Source code modified to handle .npz format
# - Map data not available in .npz (acceptable for training)

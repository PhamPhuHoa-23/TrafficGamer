# Pre-Processed Data Loading Guide

## 📋 Overview

Bạn đã xử lý xong train và val data thành `.npz` files. Document này hướng dẫn cách:
1. Upload và load pre-processed data trên Kaggle
2. Merge multiple batches
3. Hiểu về .npz vs .pkl format

---

## 🚀 Quick Start

### Upload Pre-Processed Data to Kaggle

**Step 1: Prepare data locally**
```bash
# Zip your processed data
cd /path/to/your/data
zip -r processed_train.zip processed_data/
zip -r processed_val.zip processed_data/

# Optional: Include processed log
zip processed_train.zip processed_scenarios.txt
```

**Step 2: Upload to Kaggle**
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `processed_train.zip` and/or `processed_val.zip`
4. Name it (e.g., "av2-processed-train-v1")
5. Publish dataset

**Step 3: Use in notebook**
```python
# In qcnet_argoverse.py notebook:

# Set paths
PREPROCESSED_INPUT = "/kaggle/input/av2-processed-train-v1/processed_data/"
USE_PREPROCESSED = True

# Run Section 6B to load
# Data will be loaded into `preprocessed_data` dict
```

---

## 🔀 Merging Multiple Batches

Nếu bạn đã chạy nhiều lần trên Kaggle (incremental processing), bạn có multiple batches cần merge.

### Method 1: Using Existing Script

Repository đã có sẵn `merge_processed_batches.py`:

```bash
# Locally
python merge_processed_batches.py \
    --input_dirs batch1/processed_data/ batch2/processed_data/ batch3/processed_data/ \
    --output merged_data/

# Verify
ls merged_data/*.npz | wc -l

# Zip and upload
zip -r merged_data.zip merged_data/
```

### Method 2: In Kaggle Notebook

```python
# Add all batch datasets as inputs to notebook
BATCH_DIRS = [
    "/kaggle/input/batch1/processed_data/",
    "/kaggle/input/batch2/processed_data/",
    "/kaggle/input/batch3/processed_data/",
]

# Run merge script from TrafficGamer repo
!python merge_processed_batches.py \
    --input_dirs {' '.join(BATCH_DIRS)} \
    --output /kaggle/working/merged_data/

# Use merged data
PREPROCESSED_INPUT = "/kaggle/working/merged_data/"
USE_PREPROCESSED = True
```

---

## 📦 NPZ vs PKL Format

### What's in your .npz files?

```python
# Load a sample
data = np.load('scenario_123.npz')
print(data.files)  # ['agent_hist', 'gt_future', 'scenario_path']

agent_hist = data['agent_hist']  # (50, 5) - [x, y, vx, vy, heading]
gt_future = data['gt_future']    # (60, 2) - [x, y]
```

**Limitation**: NO map data (lanes, crosswalks, traffic signals)

### What's in QCNet's .pkl files?

```python
# Full processed data
pkl_data = {
    'scenario_id': str,
    'city': str,
    'agent': {
        'position': torch.Tensor,
        'heading': torch.Tensor,
        'velocity': torch.Tensor,
        'valid_mask': torch.Tensor,
        # ... more agent features
    },
    'map_polygon': {
        'position': torch.Tensor,
        'type': torch.Tensor,
        # ... map polygon features
    },
    'map_point': {
        'position': torch.Tensor,
        'type': torch.Tensor,
        # ... map point features
    }
}
```

**Includes**: Full map data required for accurate prediction

---

## 🤔 Do You Need to Change Source Code?

### TL;DR:
- ✅ **No changes needed** if using .npz for manual evaluation (Section 12)
- ❌ **Changes needed** if using .npz with val.py (Section 10)

### Detailed Comparison:

| Use Case | .npz Files | Source Changes | Accuracy |
|----------|-----------|----------------|----------|
| Manual metrics | ✅ Works | None | Lower (no map) |
| val.py evaluation | ❌ Won't work | Need full .pkl | Full accuracy |
| Quick testing | ✅ Perfect | None | Good enough |
| Official benchmark | ❌ Not supported | Need reprocess | Required |

### Why .npz doesn't work with val.py?

```python
# In datasets/argoverse_v2_dataset.py
def process(self):
    # This expects full parquet + map data
    for raw_file_name in self.raw_file_names:
        df = pd.read_parquet(...)  # Agent data
        map_api = ArgoverseStaticMap.from_json(...)  # MAP DATA!
        
        data = {
            'agent': self.get_agent_features(df),
            **self.get_map_features(map_api)  # Needs map!
        }
```

Your .npz files were created with simplified processing (no map).

---

## 🎯 Recommendations

### For Quick Metrics (Your Current Use Case):
✅ **Use .npz files with manual evaluation**
- Fast to load
- Good enough for development/debugging
- No source code changes needed

```python
# In notebook
USE_PREPROCESSED = True
PREPROCESSED_INPUT = "/kaggle/input/your-data/processed_data/"

# Run manual evaluation (Section 12)
# Get minADE, minFDE, MR metrics
```

### For Official Evaluation:
❌ **Need full processing with map data**
- Process with `val.py` script (Section 10)
- Creates .pkl files automatically
- Includes map data
- Accurate predictions

```python
# This requires full parquet + map data
!python val.py --model QCNet --root {AV2_ROOT} --ckpt_path {CKPT_PATH}
```

---

## 🛠️ Advanced: Convert .npz to .pkl

**⚠️ Not Recommended** - You'll still be missing map data!

But if you really want to:

```python
# Already implemented in Section 6B
convert_npz_to_pkl_cache(
    npz_dir="/kaggle/input/your-npz/",
    output_dir="/kaggle/working/processed_val/"
)

# Result: .pkl files WITHOUT map data (suboptimal)
```

---

## 📊 Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│ You have: .npz files (agent history + GT)              │
└─────────────────────────────────────────────────────────┘
                        │
                        ├─── Option A: Manual Evaluation
                        │    ✅ Fast, no source changes
                        │    ❌ No map data → lower accuracy
                        │    → Upload .npz → Load → Evaluate
                        │
                        └─── Option B: Official Evaluation
                             ✅ Full accuracy with map
                             ❌ Need to reprocess data
                             → Process parquet+map → .pkl → val.py
```

### Choose Your Path:

**Quick Development/Testing?**
→ Use .npz with manual evaluation

**Publishing Results/Benchmarks?**
→ Reprocess with map data → Use val.py

---

## 🔧 Troubleshooting

### "No .npz files found"
- Check path: `ls /kaggle/input/your-dataset/`
- Unzip if needed: `unzip processed_data.zip`
- Verify structure: Should be `processed_data/scenario_*.npz`

### "Invalid .npz format"
```python
# Verify your .npz files
data = np.load('scenario_123.npz')
assert 'agent_hist' in data
assert 'gt_future' in data
assert data['agent_hist'].shape == (50, 5)
assert data['gt_future'].shape == (60, 2)
```

### "val.py fails with .npz"
- Expected! val.py needs .pkl with map data
- Use manual evaluation instead
- Or reprocess with map data

---

## 📞 Need Help?

Check these sections in the notebook:
- **Section 6B**: Load pre-processed .npz
- **Section 7**: Process new data
- **Section 10**: Official evaluation (val.py)
- **Section 12**: Manual evaluation
- **Section 14B**: Merge batches

---

## Summary

✅ **You can use your .npz files!**
- Upload to Kaggle
- Set `USE_PREPROCESSED=True`
- Load instantly
- Get quick metrics

❌ **But for full accuracy:**
- Need map data
- Process with val.py
- Creates .pkl automatically

**Bottom line**: .npz is great for development, but official evaluation needs full processing with map data.

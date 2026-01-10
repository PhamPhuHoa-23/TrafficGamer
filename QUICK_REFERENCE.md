# 🚀 Quick Reference - Pre-Processed Data

## 📋 Cheat Sheet

### Upload & Use (3 Steps)

```bash
# 1. Zip locally
zip -r processed_data.zip processed_data/

# 2. Upload to Kaggle (create new dataset)

# 3. Use in notebook
```

```python
PREPROCESSED_INPUT = "/kaggle/input/your-dataset/processed_data/"
USE_PREPROCESSED = True
# Run Section 6B
```

---

## 🔀 Merge Batches

```bash
python merge_processed_batches.py \
    --input_dirs batch1/ batch2/ batch3/ \
    --output merged/
```

---

## 📊 NPZ vs PKL

| Feature | .npz | .pkl |
|---------|------|------|
| Map data | ❌ | ✅ |
| Speed | ⚡ Fast | 🐌 Slow |
| Manual eval | ✅ | ✅ |
| val.py | ❌ | ✅ |

**When to use:**
- Development → .npz
- Official eval → .pkl

---

## 🎯 Quick Decisions

**Q: Tôi có sẵn .npz files, làm sao load?**
```python
USE_PREPROCESSED = True
PREPROCESSED_INPUT = "/kaggle/input/data/processed_data/"
```

**Q: Tôi cần merge nhiều batches?**
```bash
python merge_processed_batches.py --input_dirs batch1/ batch2/ --output merged/
```

**Q: .npz có thể dùng với val.py không?**
❌ Không - val.py cần .pkl với map data

**Q: Làm sao có .pkl files?**
Process lại với map data:
```bash
python val.py --model QCNet --root /path/to/data --ckpt_path checkpoint.ckpt
```

---

## 📂 File Structure

```
/kaggle/input/
└── your-dataset/
    └── processed_data/
        ├── scenario_001.npz
        ├── scenario_002.npz
        └── ...
```

Each .npz contains:
- `agent_hist` (50, 5) - history
- `gt_future` (60, 2) - future
- `scenario_path` - original path

---

## ⚡ Common Commands

**Verify .npz:**
```python
import numpy as np
data = np.load('scenario_001.npz')
print(data.files)
print(data['agent_hist'].shape)  # (50, 5)
print(data['gt_future'].shape)   # (60, 2)
```

**Count scenarios:**
```bash
ls processed_data/*.npz | wc -l
```

**Zip for upload:**
```bash
zip -r data.zip processed_data/
```

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| Files not found | Check path, unzip if needed |
| Wrong shape | Verify .npz format |
| val.py fails | Use manual eval OR reprocess |
| Duplicates | Use merge script with verify |

---

## 📖 Full Docs

- Detailed guide: `PREPROCESSED_DATA_GUIDE.md`
- Example config: `example_preprocessed_config.py`
- Changes log: `CHANGES_SUMMARY.md`

---

## 💡 Pro Tips

1. **Upload incrementally** - Process in batches, merge later
2. **Keep processed log** - Track what's been processed
3. **Verify before upload** - Check shapes and formats
4. **Use descriptive names** - av2-train-batch1-v1, etc.
5. **Save merge summary** - Easier to track versions

---

## 🎓 Example Workflow

```python
# ====== In Kaggle Notebook ======

# First time - process some data
RUN_BATCH_PROCESSING = True
MAX_SCENARIOS = 1000
# ... processes and saves to processed_data/

# Download processed_data/ and processed_scenarios.txt

# ====== Next run ======

# Upload previous processed_data as Kaggle dataset
# Then:

PREPROCESSED_INPUT = "/kaggle/input/batch1/processed_data/"
USE_PREPROCESSED = True

# Continue processing remaining scenarios
RUN_BATCH_PROCESSING = True
MAX_SCENARIOS = 1000  # Next 1000

# ====== Final run ======

# Merge all batches
MERGE_BATCHES = True
BATCH_DIRS = [
    "/kaggle/input/batch1/processed_data/",
    "/kaggle/input/batch2/processed_data/",
    "/kaggle/input/batch3/processed_data/",
]

# Use merged data
PREPROCESSED_INPUT = "/kaggle/working/merged_data/"

# Run evaluation
# ...
```

---

## 📞 Questions?

Check:
1. Notebook Section 6B
2. Notebook Section 📌
3. `PREPROCESSED_DATA_GUIDE.md`
4. `example_preprocessed_config.py`

# How to Retrain Your YOLOv8 Model

You have **two options** for retraining: start fresh or resume from a checkpoint.

---

## Option 1: Start Fresh Training (Recommended)

**Use this when:**
- You want to train from scratch with the pretrained `yolov8m-seg.pt` checkpoint
- You've updated your dataset or hyperparameters
- You want a clean new run

### Steps:

1. **Open the notebook** (`sweetpotato_yolov8_training.ipynb`)

2. **Run cells sequentially** up to the training cell:
   - Setup & Installation (cells 1-4)
   - Data Preparation (cells 5-9)
   - Model Training → **"Load training configuration"** cell
   - Model Training → **"Initialize YOLOv8 segmentation model"** cell

3. **In the training cell**, make sure `resume=False` (it's already set):
   ```python
   resume=False,  # Set to False for fresh training
   ```

4. **Optional: Change run name** to keep old runs separate:
   ```python
   name='sweetpotato_exp_v2',  # Change this to create a new folder
   ```

5. **Run the training cell** — it will:
   - Start from `yolov8m-seg.pt` (pretrained)
   - Train for the number of epochs in `config.yaml` (default: 100)
   - Save checkpoints every 10 epochs
   - Save results to: `sweetpotato_project/runs/segment/sweetpotato_exp/` (or your custom name)

---

## Option 2: Resume from Checkpoint

**Use this when:**
- Training was interrupted and you want to continue
- You want to train for more epochs starting from your best model

### Steps:

1. **Find your checkpoint**:
   - **Last epoch:** `runs/segment/sweetpotato_exp/weights/last.pt`
   - **Best model:** `runs/segment/sweetpotato_exp/weights/best.pt`

2. **In the training cell**, change:
   ```python
   resume='runs/segment/sweetpotato_exp/weights/last.pt',  # Path to checkpoint
   ```
   Or use the best model:
   ```python
   resume='runs/segment/sweetpotato_exp/weights/best.pt',
   ```

3. **Update epochs** if you want more training:
   - In `config.yaml`: `epochs: 150` (if you already did 100, set to 150 for 50 more)
   - Or in the notebook's `default_config`: `'epochs': 150`

4. **Run the training cell** — it will:
   - Load weights from the checkpoint
   - Continue training from that epoch
   - Save new checkpoints (will overwrite if same run name)

---

## Quick Reference

### Before Retraining — Check These:

| Item | Where | What to Check |
|------|-------|---------------|
| **Dataset** | `data.yaml` | Paths are correct, train/val/test exist |
| **Epochs** | `config.yaml` | Set to desired number (e.g., 100) |
| **Batch size** | `config.yaml` | 16 (reduce to 8 or 4 if OOM) |
| **Model** | `config.yaml` | `model: yolov8m-seg.pt` |
| **Run name** | Training cell | `name='sweetpotato_exp'` (change for new run) |
| **Resume** | Training cell | `resume=False` (fresh) or path to `.pt` (resume) |

### Where Results Are Saved:

**Fresh training:**
```
sweetpotato_project/runs/segment/sweetpotato_exp/
├── weights/
│   ├── best.pt      ← Best model (use this!)
│   ├── last.pt      ← Last epoch
│   └── epoch10.pt, epoch20.pt, ...  ← Checkpoints every 10 epochs
├── results.png      ← Loss curves
├── confusion_matrix.png
└── ...
```

**If you change `name='sweetpotato_exp_v2'`:**
```
sweetpotato_project/runs/segment/sweetpotato_exp_v2/
└── (same structure)
```

---

## Common Scenarios

### Scenario 1: "I want to train more epochs"

**Option A — Resume from best:**
```python
resume='runs/segment/sweetpotato_exp/weights/best.pt',
epochs=150,  # In config.yaml or default_config
```

**Option B — Fresh start:**
```python
resume=False,
epochs=150,  # In config.yaml
name='sweetpotato_exp_150epochs',  # New run name
```

### Scenario 2: "I updated my dataset"

1. Make sure your new dataset is in the correct location
2. Update `data.yaml` paths if needed
3. Run data validation cells
4. Start fresh training:
   ```python
   resume=False,
   name='sweetpotato_exp_v2',  # New run name
   ```

### Scenario 3: "Training crashed, want to continue"

1. Find your last checkpoint: `weights/last.pt` or `weights/epochXX.pt`
2. Resume:
   ```python
   resume='runs/segment/sweetpotato_exp/weights/last.pt',
   # Keep same epochs (will continue from where it stopped)
   ```

### Scenario 4: "I want to try different hyperparameters"

1. Edit `config.yaml` (e.g., change `lr0`, `batch`, `epochs`)
2. Start fresh:
   ```python
   resume=False,
   name='sweetpotato_exp_lr0.001',  # Descriptive name
   ```

---

## Tips

- **Keep old runs:** Change `name=` to create a new folder, so old results aren't overwritten
- **Monitor GPU:** If you get OOM, reduce `batch` in `config.yaml` (16 → 8 → 4)
- **Checkpoints:** Saved every 10 epochs (`save_period: 10` in config)
- **Best model:** Always use `best.pt` for inference (it has the highest validation mAP)

---

## Example: Fresh Retrain with New Run Name

```python
# In the training cell, change:
name='sweetpotato_exp_retrain_2024',  # New name
resume=False,  # Fresh start
# Keep everything else the same
```

Then run the cell. Results will be in:
```
sweetpotato_project/runs/segment/sweetpotato_exp_retrain_2024/
```

Your old run (`sweetpotato_exp`) will remain untouched.

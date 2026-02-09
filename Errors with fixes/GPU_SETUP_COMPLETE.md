# Complete GPU Setup Guide

## ✅ What Has Been Fixed

1. **Cell 16**: Enhanced GPU diagnostics with device setup
2. **Training Cell**: Will be updated to use GPU automatically

## 📋 STEP-BY-STEP CHECKLIST

### Step 1: Install CUDA-Enabled PyTorch (TERMINAL)

**Open PowerShell or Command Prompt and run:**

```bash
# Uninstall CPU-only PyTorch
pip uninstall -y torch torchvision torchaudio

# Install CUDA-enabled PyTorch (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

**Alternative CUDA versions if 12.1 doesn't work:**
- CUDA 11.8: `pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
- CUDA 12.4: `pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio`

### Step 2: Restart Kernel

In Jupyter/Cursor:
- **Kernel → Restart** (or restart the notebook)

### Step 3: Verify GPU Detection

**Run Cell 16** (GPU Diagnostics). You should see:
- ✓ PyTorch has CUDA support (no `+cpu`)
- ✓ CUDA Available: True
- ✓ GPU: [Your GPU name]
- ✓ Device Selected: cuda

### Step 4: Update Training Cell

**Find the training cell** (the one with `model.train(`) and **add this code BEFORE `model.train(`:**

```python
# Ensure GPU is used if available
if 'TRAINING_DEVICE' in globals():
    train_device = TRAINING_DEVICE
elif 'device' in globals() and hasattr(device, 'type') and device.type == 'cuda':
    train_device = '0'  # Use first GPU
elif torch.cuda.is_available():
    train_device = '0'  # Use first GPU
else:
    train_device = 'cpu'  # Fallback to CPU

# For YOLO, use '0' for first GPU, '1' for second, etc., or 'cpu'
if train_device == 'cuda' and torch.cuda.is_available():
    train_device = '0'  # Convert 'cuda' to '0' for YOLO
    print(f"✓ Training on GPU {train_device}: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif train_device == '0' and torch.cuda.is_available():
    print(f"✓ Training on GPU {train_device}: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"⚠ Training on CPU (GPU not available)")
    print("  Install CUDA-enabled PyTorch to use GPU")
```

**Then change the `model.train(` call to include `device`:**

**BEFORE:**
```python
results = model.train(
    data=data_yaml_path,
    epochs=default_config['epochs'],
    ...
)
```

**AFTER:**
```python
results = model.train(
    data=data_yaml_path,
    device=train_device,  # Add this line - '0' for GPU, 'cpu' for CPU
    epochs=default_config['epochs'],
    ...
)
```

## 🎯 Quick Reference: Code Changes

### Device Setup (Already in Cell 16)
Cell 16 now sets:
- `device = torch.device("cuda")` if GPU available
- `TRAINING_DEVICE = 'cuda'` or `'cpu'`

### Training Cell Update Needed
Add `device=train_device` parameter to `model.train()` call.

## ✅ Verification

After setup, when training starts, you should see:
- `device=0` (or `device=cuda`) in training parameters
- `GPU_mem: X.XG` in training progress (not 0)
- Much faster training times

## 🔧 Troubleshooting

**If GPU still not detected after install:**
1. Verify: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check: `nvidia-smi` shows your GPU
3. Try: Restart computer (sometimes needed for driver changes)

**If training still uses CPU:**
- Check Cell 16 output shows GPU is available
- Verify `device=train_device` is in `model.train()` call
- Check training output shows `device=0` not `device=None`

# Fix: Why CPU is Being Used Instead of GPU

## Problem
Your training is using CPU instead of GPU because:
1. **PyTorch CPU-only version installed**: `torch-2.10.0+cpu` (see output)
2. **CUDA not available**: `CUDA available: False`
3. **No device specified**: Training uses `device=None` which defaults to CPU

## Root Cause
You have the **CPU-only version of PyTorch** installed. This version cannot use GPU even if you have one.

## Solution

### Step 1: Check if you have a GPU
Open PowerShell/Terminal and run:
```bash
nvidia-smi
```

If this works, you have an NVIDIA GPU and drivers installed.

### Step 2: Install CUDA-enabled PyTorch

**Uninstall CPU version:**
```bash
pip uninstall torch torchvision
```

**Install CUDA version (choose based on your CUDA version):**

For CUDA 12.1 (most common):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Check your CUDA version:**
```bash
nvidia-smi
```
Look for "CUDA Version: X.X" in the output.

### Step 3: Verify GPU is Available

After installing, restart your kernel and run:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

You should see:
- PyTorch version WITHOUT `+cpu` (e.g., `2.10.0+cu121`)
- `CUDA Available: True`
- Your GPU name

### Step 4: Update Training Cell

In the training cell, add `device` parameter:

**Before:**
```python
results = model.train(
    data=data_yaml_path,
    epochs=default_config['epochs'],
    ...
)
```

**After:**
```python
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

results = model.train(
    data=data_yaml_path,
    device=device,  # Add this line
    epochs=default_config['epochs'],
    ...
)
```

## Quick Fix Commands

Run these in your terminal (not in notebook):

```bash
# Uninstall CPU PyTorch
pip uninstall torch torchvision -y

# Install CUDA PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## After Fixing

1. **Restart your Jupyter kernel** (Kernel → Restart)
2. **Re-run the import cell** to verify CUDA is available
3. **Run the GPU diagnostics cell** (Cell 16) to confirm GPU is detected
4. **Run training** - it should now use GPU automatically

## Expected Output After Fix

When training, you should see:
- `device=cuda` or `device=0` in the training parameters
- GPU memory usage in the training progress (e.g., `GPU_mem: 2.1G`)
- Much faster training times

## Troubleshooting

**If nvidia-smi doesn't work:**
- Install NVIDIA drivers from: https://www.nvidia.com/drivers
- Restart your computer after installing

**If CUDA still not available after installing:**
- Check CUDA version compatibility
- Try: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` (older, more compatible)

**If you have AMD GPU:**
- PyTorch doesn't support AMD GPUs directly
- Use ROCm (AMD's CUDA alternative) - more complex setup
- Or use CPU training (slower but works)

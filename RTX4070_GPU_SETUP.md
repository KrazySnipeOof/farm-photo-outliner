# RTX 4070 GPU Setup - Complete Guide

## TERMINAL COMMANDS (Run These FIRST in Cursor Terminal)

**Open PowerShell or Command Prompt in Cursor and run:**

```bash
# Step 1: FULL UNINSTALL of CPU PyTorch
pip uninstall -y torch torchvision torchaudio ultralytics

# Step 2: Install CUDA 12.4 PyTorch (for RTX 4070 / RTX 40-series)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio --upgrade

# Step 3: Reinstall ultralytics
pip install ultralytics --upgrade

# Step 4: Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output after verification:**
- PyTorch: `2.x.x+cu124` (NOT `+cpu`)
- CUDA: `True`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU` (or similar)

## REPLACEMENT CELLS (Already Updated in Notebook)

### ✅ Cell 16: Enhanced GPU Diagnostics
**Status:** ✅ Already updated - will show RTX 4070 detection

### ✅ Cell 17: Auto-Fix Installation
**Status:** ✅ Already added - automatically installs GPU PyTorch if CUDA not available

### ✅ Cell 20: Training with GPU
**Status:** ✅ Already updated - forces `device=0` for GPU

## CHECKLIST

### Step 1: Install CUDA PyTorch (Terminal)
- [ ] Run uninstall command
- [ ] Run install command (CUDA 12.4)
- [ ] Run ultralytics reinstall
- [ ] Verify with Python command above

### Step 2: Restart Kernel
- [ ] In Cursor: **Ctrl+Shift+P → "Jupyter: Restart Kernel"**
- [ ] Or: **Kernel → Restart** in notebook
- [ ] **CRITICAL:** Must restart after PyTorch install

### Step 3: Run Diagnostics
- [ ] Run **Cell 16** (GPU Diagnostics)
- [ ] Should show: `✓ PyTorch has CUDA support` (no `+cpu`)
- [ ] Should show: `✓ CUDA available: True`
- [ ] Should show: `✓ GPU 0: NVIDIA GeForce RTX 4070 Laptop GPU`
- [ ] Should show: `Memory: 8.0 GB`
- [ ] Should show: `✅ RTX 4070 detected!`

### Step 4: Run Training
- [ ] Run **Cell 20** (Training)
- [ ] Should show: `✓ Training on GPU 0: NVIDIA GeForce RTX 4070`
- [ ] Should show: `✅ RTX 4070 detected and ready!`
- [ ] Training output should show: `device=0` (not `device=None`)
- [ ] Training progress should show: `GPU_mem: X.XG` (not 0)

## TROUBLESHOOTING CHECKLIST

### ✅ NVIDIA Drivers
- [ ] NVIDIA drivers >= 551.xx installed (check Device Manager)
- [ ] `nvidia-smi` works in terminal
- [ ] RTX 4070 shows in Device Manager

### ✅ CUDA Toolkit
- [ ] CUDA Toolkit 12.4 installed? (Usually NOT needed - PyTorch wheels include CUDA)
- [ ] If missing, download from: https://developer.nvidia.com/cuda-downloads
- [ ] **Note:** PyTorch wheels usually include CUDA, so toolkit may not be required

### ✅ Python Environment
- [ ] Using correct conda/virtual environment
- [ ] Python 3.8+ (required for PyTorch)
- [ ] No conflicting PyTorch installations

### ✅ After Installation
- [ ] **Restart Cursor/Python kernel** (CRITICAL!)
- [ ] Re-run Cell 16 to verify GPU detection
- [ ] Check PyTorch version shows `+cu124` not `+cpu`

## Expected Output After Fix

**Cell 16 Output:**
```
PyTorch: 2.10.0+cu124
✓ PyTorch has CUDA support
CUDA available: True
✓ CUDA version: 12.4
✓ GPU count: 1

✓ GPU 0: NVIDIA GeForce RTX 4070 Laptop GPU
  Memory: 8.0 GB
  Compute Capability: 8.9
  ✅ RTX 4070 detected!

✓ Device Selected: cuda
✓ Training will use: GPU 0 (NVIDIA GeForce RTX 4070 Laptop GPU)
```

**Cell 20 Output (Training):**
```
======================================================================
TRAINING CONFIGURATION - GPU MODE
======================================================================
✓ Training on GPU 0: NVIDIA GeForce RTX 4070 Laptop GPU
✓ GPU Memory: 8.0 GB
✅ RTX 4070 detected and ready!
======================================================================
...
device=0  (in training parameters)
GPU_mem: X.XG  (in training progress)
```

## Alternative: Use Auto-Fix Cell

If terminal commands don't work, you can use **Cell 17 (Auto-Fix)**:
1. Run Cell 17
2. It will automatically uninstall CPU PyTorch and install CUDA 12.4
3. **Restart kernel** after it completes
4. Re-run Cell 16 to verify

## Why CUDA 12.4?

- RTX 4070 is part of RTX 40-series (Ada Lovelace architecture)
- CUDA 12.4 is the latest stable version compatible with RTX 40-series
- PyTorch 2.6.0+ supports CUDA 12.4
- Better performance and compatibility than older CUDA versions

## Verification Commands

After installation, verify with:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

You should see:
- `PyTorch: 2.x.x+cu124` (NOT `+cpu`)
- `CUDA Available: True`
- `GPU: NVIDIA GeForce RTX 4070 Laptop GPU`
- `Memory: 8.0 GB`

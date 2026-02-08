# Install CUDA-Enabled PyTorch for GPU Training

## Quick Fix Commands

**Run these commands in your terminal (PowerShell or Command Prompt):**

```bash
# Step 1: Uninstall CPU-only PyTorch
pip uninstall -y torch torchvision torchaudio

# Step 2: Install CUDA-enabled PyTorch (CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## Alternative CUDA Versions

If CUDA 12.1 doesn't work, try these based on your NVIDIA driver version:

**For CUDA 11.8:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

**For CUDA 12.4:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## Check Your CUDA Version

Run this to see your CUDA version:
```bash
nvidia-smi
```

Look for "CUDA Version: X.X" in the output.

## After Installation

1. **Restart your Jupyter kernel** (Kernel → Restart)
2. **Re-run Cell 16** (GPU Diagnostics) to verify GPU is detected
3. **Run training** - it should now use GPU automatically

## Verify Installation

After installing, verify with:
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

## Troubleshooting

**If installation fails:**
- Make sure you're in the correct conda/virtual environment
- Try: `pip install --upgrade pip` first
- Check Python version compatibility (PyTorch requires Python 3.8+)

**If CUDA still not available after install:**
- Verify NVIDIA drivers: `nvidia-smi` should work
- Check CUDA toolkit is installed (separate from PyTorch)
- Try restarting your computer

# Fix NumPy 2.x Compatibility Error

## Problem
If you see this error:
```
AttributeError: _ARRAY_API not found
```

This means NumPy 2.x is installed, but `opencv-python` was compiled for NumPy 1.x and is incompatible.

## Solution (Choose One)

### Option 1: Run Fix Script (Easiest)
**Before opening the notebook**, run this in your terminal:

```bash
python fix_numpy.py
```

Or manually:
```bash
pip install 'numpy<2.0' --force-reinstall
```

### Option 2: Fix in Notebook
1. **Restart the kernel** (Kernel → Restart)
2. Run **Cell 3** (the NumPy fix cell) FIRST
3. **Restart the kernel again** after the fix
4. Then run Cell 4 (Installation) and Cell 5 (Imports)

### Option 3: Manual Terminal Fix
Open PowerShell/Terminal and run:
```bash
pip install 'numpy<2.0' opencv-python --force-reinstall
```

Then restart your Jupyter kernel.

## Why This Happens
- NumPy 2.x removed the `_ARRAY_API` that opencv-python needs
- opencv-python was compiled for NumPy 1.x
- The fix: downgrade NumPy to < 2.0

## Verify Fix
After fixing, verify with:
```python
import numpy
print(numpy.__version__)  # Should show 1.x.x, not 2.x.x
```

## Important Notes
- **Always restart the kernel** after downgrading NumPy
- The fix cell (Cell 3) checks version without importing NumPy (safer)
- If the kernel crashes, fix NumPy from terminal first, then restart

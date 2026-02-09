# YOLOv8 Model Loading RecursionError – Fix Summary

## What Caused the Recursion

- You had a **custom `torch.load` patch** (PyTorch 2.6+ `weights_only` fix) that did:
  - `_original_torch_load = torch.load`
  - `torch.load = _patched_torch_load`
  - `_patched_torch_load` called `_original_torch_load(...)`.
- By the time this ran, **Ultralytics had already** replaced `torch.load` with its own wrapper in `ultralytics.utils.patches`. So `_original_torch_load` was **Ultralytics’ wrapper**, not the real PyTorch loader.
- Ultralytics’ wrapper does `torch.load(...)` again. After your patch, `torch.load` is your `_patched_torch_load`, so the call chain became:
  - `torch.load()` → `_patched_torch_load` → `_original_torch_load` (Ultralytics) → `torch.load()` → `_patched_torch_load` → … → **infinite recursion** → `RecursionError`.

So the recursion came from your patch calling whatever was bound to `torch.load` at patch time (Ultralytics’ wrapper), which in turn called `torch.load` again.

## How the New Code Prevents It

- The patch no longer uses `torch.load` or whatever is currently bound to it. It uses the **real** loader that never gets patched:
  - `_real_torch_load = getattr(torch.serialization, "load", torch.load)`  
  so we call `torch.serialization.load` (or fallback to `torch.load` only if `load` is missing).
- `_patched_torch_load` now calls **only** `_real_torch_load(...)`, so there is no second `torch.load` call and no loop.
- Model init is unchanged in behavior: we still call `YOLO(MODEL_NAME)` once at the top level, with a safe check on the model spec.

## BEFORE (relevant block)

```python
# Patch torch.load to use weights_only=False for YOLOv8 compatibility
_original_torch_load = torch.load   # ← This was Ultralytics’ wrapper!

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)   # ← Calls torch.load again → recursion

torch.load = _patched_torch_load
# ...
model_name = default_config['model']
model = YOLO(model_name)
```

## AFTER (relevant block)

```python
# Use torch.serialization.load so we never re-enter torch.load
_real_torch_load = getattr(torch.serialization, "load", torch.load)

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)   # ← Single real load, no recursion

torch.load = _patched_torch_load
# ...
MODEL_NAME = default_config.get("model", "yolov8m-seg.pt")
MODEL_NAME = str(MODEL_NAME).strip() if hasattr(MODEL_NAME, "strip") else str(MODEL_NAME)

suffix = Path(MODEL_NAME).suffix.lower()
if suffix in (".pt", ".onnx", ".engine", ".yaml") or (suffix == "" and MODEL_NAME.strip()):
    model = YOLO(MODEL_NAME)
else:
    raise ValueError(f"Invalid model spec: {MODEL_NAME!r} – expected a weights/config path or YOLO name (e.g. yolov8m-seg.pt).")

print(f"✓ Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
```

## Config / `default_config["model"]`

- `default_config["model"]` is set from `config.yaml` (or notebook defaults) and is the **string** `'yolov8m-seg.pt'`. No YOLO or Model instance is stored there.
- The new code uses `default_config.get("model", "yolov8m-seg.pt")`, normalizes it to a string, and validates it with the `Path(MODEL_NAME).suffix` check before calling `YOLO(MODEL_NAME)` once.

## What Stayed the Same

- `model.train(...)` and `model.predict(...)` are unchanged.
- Dataset paths, `data.yaml`, and logging are unchanged.
- Only the model-loading cell was changed: recursion-safe patch + single top-level `YOLO(MODEL_NAME)` and safe model-spec check.

Running the updated cell should load `yolov8m-seg.pt` once, avoid `RecursionError`, and print the model parameter count.

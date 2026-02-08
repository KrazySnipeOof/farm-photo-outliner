#!/usr/bin/env python
"""
Quick fix script for NumPy 2.x / opencv-python compatibility issue
Run this script BEFORE running the notebook if you see _ARRAY_API errors

Usage:
    python fix_numpy.py
"""

import subprocess
import sys

print("="*70)
print("FIXING NUMPY 2.X COMPATIBILITY ISSUE")
print("="*70)

# Check current NumPy version
print("\n1. Checking NumPy version...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'show', 'numpy'],
    capture_output=True,
    text=True,
    timeout=30
)

if result.returncode == 0:
    numpy_version = None
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            numpy_version = line.split(':', 1)[1].strip()
            break
    
    if numpy_version:
        major_version = int(numpy_version.split('.')[0])
        print(f"   Current NumPy version: {numpy_version}")
        
        if major_version >= 2:
            print(f"\n⚠ PROBLEM: NumPy {numpy_version} is incompatible with opencv-python")
            print("   opencv-python requires NumPy < 2.0")
            
            print("\n2. Downgrading NumPy to < 2.0...")
            fix_result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'numpy<2.0', '--force-reinstall'],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if fix_result.returncode == 0:
                # Verify
                verify_result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', 'numpy'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if verify_result.returncode == 0:
                    for line in verify_result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            new_version = line.split(':', 1)[1].strip()
                            print(f"\n✓ SUCCESS: NumPy downgraded to {new_version}")
                            print("\n✅ Fix complete! You can now run the notebook.")
                            sys.exit(0)
                print("\n✓ NumPy downgrade completed")
            else:
                print(f"\n✗ Failed to downgrade NumPy")
                if fix_result.stderr:
                    print(f"   Error: {fix_result.stderr[:300]}")
                print(f"\n   Try manually:")
                print(f"   {sys.executable} -m pip install 'numpy<2.0' --force-reinstall")
                sys.exit(1)
        else:
            print(f"\n✓ NumPy {numpy_version} is already compatible (< 2.0)")
            print("   No fix needed!")
            sys.exit(0)
    else:
        print("   ⚠ Could not determine NumPy version")
else:
    print("   NumPy not installed - will be installed correctly by notebook")
    print("   Pre-installing NumPy < 2.0...")
    pre_result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'numpy<2.0'],
        capture_output=True,
        text=True,
        timeout=120
    )
    if pre_result.returncode == 0:
        print("   ✓ NumPy < 2.0 pre-installed")
    else:
        print("   ⚠ Pre-installation failed")

print("="*70)

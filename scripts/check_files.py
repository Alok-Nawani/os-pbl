#!/usr/bin/env python3
"""Check if required files exist and provide guidance."""

import os

required_files = [
    "adaptive_os_simulator/__init__.py",
    "adaptive_os_simulator/gui/__init__.py",
    "adaptive_os_simulator/gui/main_window.py",
    "adaptive_os_simulator/backend/__init__.py",
    "adaptive_os_simulator/backend/core.py",
    "adaptive_os_simulator/backend/schedulers.py",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("Checking for required files...")
print("=" * 60)

missing = []
for file in required_files:
    path = os.path.join(project_root, file)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {file}")
    if not exists:
        missing.append(file)

print("=" * 60)

if missing:
    print(f"\n❌ {len(missing)} file(s) missing!")
    print("\n📝 TO FIX:")
    print("1. In Cursor, open each missing file")
    print("2. Press Cmd+S (Mac) or Ctrl+S (Windows/Linux) to save")
    print("3. Or use 'File > Save All'")
    print("\nMissing files:")
    for f in missing:
        print(f"   - {f}")
    print("\nAfter saving, run: python3 scripts/run_gui.py")
else:
    print("\n✓ All required files exist!")
    print("You can now run: python3 scripts/run_gui.py")


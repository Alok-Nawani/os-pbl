#!/usr/bin/env python3
"""
Launch the OS Load Simulator GUI application.
This script will attempt to find and load the GUI module.
"""

import sys
import os
import importlib.util

def find_main_window():
    """Find main_window.py in various possible locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    workspace_root = os.path.dirname(project_root)
    
    possible_paths = [
        os.path.join(project_root, "adaptive_os_simulator", "gui", "main_window.py"),
        os.path.join(workspace_root, "Adaptive-CPU-Scheduler", "adaptive_os_simulator", "gui", "main_window.py"),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    return None

def main():
    """Launch the GUI."""
    print("=" * 70)
    print("OS Load Simulator - GUI Launcher")
    print("=" * 70)
    
    main_window_path = find_main_window()
    
    if not main_window_path:
        print("\n❌ ERROR: Could not find main_window.py")
        print("\nThe adaptive_os_simulator directory doesn't exist on disk.")
        print("This usually means files are open in your editor but not saved.")
        print("\n📝 SOLUTION:")
        print("1. In Cursor, press Cmd+S (or Ctrl+S) to save the current file")
        print("2. Or use 'File > Save All' to save all open files")
        print("3. Make sure these directories exist:")
        print("   - Adaptive-CPU-Scheduler/adaptive_os_simulator/")
        print("   - Adaptive-CPU-Scheduler/adaptive_os_simulator/gui/")
        print("   - Adaptive-CPU-Scheduler/adaptive_os_simulator/backend/")
        print("\nAfter saving, run this script again.")
        print("=" * 70)
        sys.exit(1)
    
    print(f"✓ Found main_window.py at: {main_window_path}")
    
    # Add parent directories to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(main_window_path)))
    sys.path.insert(0, project_root)
    
    try:
        # Try normal import first
        from PyQt6.QtWidgets import QApplication
        from adaptive_os_simulator.gui.main_window import MainWindow
        
        print("✓ Successfully imported modules")
        print("=" * 70)
        print("Launching GUI window...")
        print("=" * 70)
        
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        window = MainWindow()
        window.show()
        
        print("✓ GUI window opened successfully!")
        print("=" * 70)
        
        sys.exit(app.exec())
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nTrying alternative loading method...")
        
        # Try direct file loading
        try:
            spec = importlib.util.spec_from_file_location("main_window", main_window_path)
            main_window = importlib.util.module_from_spec(spec)
            
            # Set up module structure
            backend_dir = os.path.join(os.path.dirname(os.path.dirname(main_window_path)), "backend")
            if os.path.exists(backend_dir):
                sys.path.insert(0, os.path.dirname(backend_dir))
            
            spec.loader.exec_module(main_window)
            MainWindow = main_window.MainWindow
            
            from PyQt6.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            window = MainWindow()
            window.show()
            
            print("✓ GUI launched using direct file loading!")
            sys.exit(app.exec())
            
        except Exception as e2:
            print(f"\n❌ Failed to load: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Installation script for Amazon Archaeological Discovery Pipeline
This script handles the installation process correctly by running setup.py with proper arguments
"""

import os
import sys
import subprocess

def main():
    print("Installing Amazon Archaeological Discovery Pipeline...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run setup.py install
    try:
        subprocess.check_call([sys.executable, "setup.py", "install"])
        print("\nInstallation completed successfully!")
        print("You can now use the Amazon Archaeological Discovery Pipeline.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

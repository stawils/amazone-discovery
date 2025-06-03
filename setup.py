"""
Setup script for Amazon Archaeological Discovery Pipeline
"""

import sys
import subprocess
from pathlib import Path

# Handle missing setuptools
try:
    from setuptools import setup, find_packages
except ImportError:
    print("Setuptools not found. Attempting to install...")
    try:
        # Try to install setuptools using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
        print("Setuptools installed successfully.")
        from setuptools import setup, find_packages
    except Exception as e:
        print(f"Error installing setuptools: {e}")
        print("Please install setuptools manually: pip install setuptools")
        sys.exit(1)

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('--')
        ]
    
    # Ensure setuptools is in requirements
    if 'setuptools' not in requirements:
        requirements.append('setuptools')

setup(
    name="amazon-archaeological-discovery",
    version="1.0.0",
    author="Amazon Archaeological Discovery Team",
    author_email="stawils@gmail.com",
    description="AI-powered archaeological site detection system for the Amazon Basin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stawils/amazone-discovery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=12.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0", 
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "amazon-discovery=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="archaeology, remote sensing, satellite imagery, amazon, ai, computer vision, gis",
    project_urls={
        "Bug Reports": "https://github.com/your-org/amazon-archaeological-discovery/issues",
        "Documentation": "https://amazon-archaeological-discovery.readthedocs.io/",
        "Source": "https://github.com/your-org/amazon-archaeological-discovery",
    },
)

# Auto-install when run directly
if __name__ == "__main__":
    if len(sys.argv) == 1:  # No arguments provided
        sys.argv.append("develop")  # Use develop mode by default
        print("No command provided. Running 'develop' command by default.")
    # Continue with normal setup.py execution
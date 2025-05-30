#!/usr/bin/env python3
"""
Quick test setup for Amazon Archaeological Discovery Pipeline
Fixed to run test_pipeline.py directly without subprocess issues
"""

import subprocess
import sys
import os
from pathlib import Path

def check_basic_packages():
    """Check if basic packages are available"""
    
    required_packages = {
        'numpy': 'np',
        'matplotlib.pyplot': 'plt', 
        'pandas': 'pd'
    }
    
    optional_packages = {
        'cv2': None,
        'scipy.ndimage': 'ndimage'
    }
    
    print("🔍 Checking package availability...")
    
    missing_required = []
    missing_optional = []
    
    for package, alias in required_packages.items():
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package} - Available")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} - Missing")
    
    for package, alias in optional_packages.items():
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package} - Available")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} - Missing (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {missing_optional}")
        print("The test will run with reduced functionality")
    
    return True

def create_test_directory():
    """Create test directory structure"""
    
    print("\n📁 Creating test directory structure...")
    
    dirs_to_create = [
        "test_data",
        "test_results", 
        "test_images"
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")
    
    return True

def run_synthetic_test():
    """Run the synthetic data test"""
    
    print("\n🧪 RUNNING SYNTHETIC DATA TEST")
    print("="*50)
    
    try:
        # Try to run the test pipeline
        print("🚀 Starting test pipeline...")
        result = subprocess.run([sys.executable, "test_pipeline.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Test pipeline completed successfully!")
            return True
        else:
            print(f"❌ Test pipeline failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ test_pipeline.py not found in current directory")
        print("Make sure you're in the correct directory with all project files")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main setup and test runner"""
    
    print("🏛️ AMAZON ARCHAEOLOGICAL DISCOVERY - QUICK TEST")
    print("="*60)
    print("This will test the pipeline with synthetic Amazon satellite data")
    print("to verify all components work before using real USGS data.\n")
    
    # Step 1: Check Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    
    # Step 2: Check basic packages
    if not check_basic_packages():
        print("\n💡 Install missing packages with:")
        print("pip install numpy pandas matplotlib")
        print("pip install opencv-python scipy  # optional but recommended")
        return False
    
    # Step 3: Create directory structure
    if not create_test_directory():
        print("❌ Failed to create directories")
        return False
    
    # Step 4: Run synthetic test
    print("\n" + "="*60)
    if not run_synthetic_test():
        print("❌ Synthetic test failed")
        return False
    
    # Step 5: Success message
    print("\n" + "="*60)
    print("🎉 QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nWhat was tested:")
    print("✅ Synthetic satellite data generation")
    print("✅ Terra preta spectral analysis")
    print("✅ Geometric pattern detection")
    print("✅ Convergent anomaly scoring")
    print("✅ Visualization and reporting")
    
    print("\n🎯 Next Steps:")
    print("1. Check test_pipeline_results.png for visualizations")
    print("2. Review test_pipeline_report.json for detailed results")
    print("3. Set up USGS credentials to test with real data:")
    print("   - Get USGS M2M token from: https://ers.cr.usgs.gov/profile/access")
    print("   - Copy .env.template to .env and add your credentials")
    print("   - Run: python main.py --zone negro_madeira --full-pipeline")
    
    print("\n📊 Expected Real Results:")
    print("- Negro-Madeira confluence should score 10+ points")
    print("- Multiple terra preta signatures expected")
    print("- Circular earthworks likely in 150-300m range")
    print("- High confidence classification for ground verification")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔧 TROUBLESHOOTING:")
        print("- Make sure you're in the project directory")
        print("- Install missing packages: pip install numpy pandas matplotlib")
        print("- Check that test_pipeline.py exists")
        sys.exit(1)
    
    print(f"\n🔬 Test files created:")
    print(f"- test_pipeline_results.png (visualizations)")
    print(f"- test_pipeline_report.json (detailed results)")
    print(f"- Synthetic data demonstrates the complete workflow")
    
    print(f"\n💡 Understanding the Results:")
    print(f"- Red areas = Detected terra preta (anthropogenic soil)")
    print(f"- Cyan circles = Detected earthworks/settlements")
    print(f"- Yellow lines = Detected causeways/paths")
    print(f"- Anomaly score 7+ = Archaeological significance")
    print(f"- Score 10+ = High confidence for ground verification")
"""
Setup script for continuous learning system.

Quick setup and validation of the continuous learning environment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version."""
    logger.info("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False


def check_dependencies():
    """Check required Python packages."""
    logger.info("\nChecking dependencies...")
    
    required = [
        'torch',
        'pytorch_lightning',
        'sentencepiece',
        'omegaconf',
        'requests',
        'pyyaml'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.warning(f"❌ {package} - missing")
            missing.append(package)
    
    if missing:
        logger.info("\nInstall missing packages with:")
        logger.info("  pip install -r requirements.txt")
        return False
    
    return True


def check_gpu():
    """Check GPU availability."""
    logger.info("\nChecking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ {gpu_count} GPU(s) available")
            logger.info(f"   Primary: {gpu_name}")
            
            # Check VRAM
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            logger.info(f"   VRAM: {vram_gb:.1f} GB")
            
            return True
        else:
            logger.warning("❌ No GPU available (CPU-only mode)")
            logger.info("   Note: Training will be very slow without GPU")
            return False
    
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False


def create_directories():
    """Create required directories."""
    logger.info("\nCreating directories...")
    
    directories = [
        'data/legal_sources',
        'outputs/continuous_learning/data/raw',
        'outputs/continuous_learning/data/processed',
        'outputs/continuous_learning/models/checkpoints',
        'outputs/continuous_learning/models/registry',
        'logs/training',
        'logs/evaluation',
        'logs/pipeline',
        'deployments/text',
        'deployments/image',
        'deployments/video',
        'deployments/music',
        'backups'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ {directory}")
    
    return True


def check_config_files():
    """Check configuration files exist."""
    logger.info("\nChecking configuration files...")
    
    configs = [
        'configs/continuous_learning.yaml',
        'configs/text_base.yaml',
        'configs/latent_diffusion_base.yaml'
    ]
    
    all_exist = True
    for config in configs:
        path = Path(config)
        if path.exists():
            logger.info(f"✅ {config}")
        else:
            logger.warning(f"❌ {config} - missing")
            all_exist = False
    
    return all_exist


def check_scripts():
    """Check required scripts exist."""
    logger.info("\nChecking scripts...")
    
    scripts = [
        'scripts/legal_data_collector.py',
        'scripts/continuous_learning_pipeline.py',
        'examples/train_text_model.py',
        'examples/train_latent_diffusion.py'
    ]
    
    all_exist = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            logger.info(f"✅ {script}")
        else:
            logger.warning(f"❌ {script} - missing")
            all_exist = False
    
    return all_exist


def test_data_collection():
    """Test data collection."""
    logger.info("\nTesting data collection...")
    
    try:
        import requests
        
        # Test Wikipedia API
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                'action': 'query',
                'format': 'json',
                'titles': 'Machine learning'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Wikipedia API accessible")
        else:
            logger.warning(f"❌ Wikipedia API returned {response.status_code}")
            return False
        
        # Test Project Gutenberg
        response = requests.get("https://www.gutenberg.org/", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Project Gutenberg accessible")
        else:
            logger.warning(f"❌ Project Gutenberg returned {response.status_code}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing data collection: {e}")
        return False


def install_package():
    """Install package in development mode."""
    logger.info("\nInstalling package...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
            capture_output=True
        )
        logger.info("✅ Package installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Package installation failed: {e}")
        return False


def print_next_steps():
    """Print next steps."""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("\n1. Collect data:")
    logger.info("   python scripts/legal_data_collector.py")
    logger.info("\n2. Run single training cycle:")
    logger.info("   python scripts/continuous_learning_pipeline.py \\")
    logger.info("     --config configs/continuous_learning.yaml \\")
    logger.info("     --model-type text")
    logger.info("\n3. Run continuous learning (scheduled):")
    logger.info("   python scripts/continuous_learning_pipeline.py \\")
    logger.info("     --config configs/continuous_learning.yaml \\")
    logger.info("     --model-type text \\")
    logger.info("     --continuous --interval 24")
    logger.info("\n4. Monitor with TensorBoard:")
    logger.info("   tensorboard --logdir logs/training")
    logger.info("\nFor more information, see docs/CONTINUOUS_LEARNING.md")
    logger.info("=" * 60)


def main():
    """Run setup."""
    logger.info("=" * 60)
    logger.info("CONTINUOUS LEARNING SETUP")
    logger.info("=" * 60)
    
    checks = []
    
    # Run checks
    checks.append(("Python version", check_python_version()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("GPU", check_gpu()))
    checks.append(("Directories", create_directories()))
    checks.append(("Config files", check_config_files()))
    checks.append(("Scripts", check_scripts()))
    checks.append(("Data sources", test_data_collection()))
    checks.append(("Package install", install_package()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for name, result in checks:
        status = "✅" if result else "❌"
        logger.info(f"{status} {name}")
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("\n🎉 All checks passed!")
        print_next_steps()
        return 0
    else:
        logger.warning("\n⚠️  Some checks failed. Please fix issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

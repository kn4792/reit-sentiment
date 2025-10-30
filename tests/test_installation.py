import pytest
import sys

def test_python_version():
    """Test that Python version is 3.8 or higher"""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"

def test_src_package_importable():
    """Test that src package can be imported"""
    try:
        import src
        assert True
    except ImportError:
        pytest.fail("src package not importable. Run 'pip install -e .' first")

def test_required_packages():
    """Test that core packages are installed"""
    required_packages = ['pandas', 'numpy', 'pytest']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"{package} is not installed. Run 'pip install -r requirements.txt'")

def test_project_structure():
    """Test that basic project structure exists"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = ['src', 'tests', 'scripts', 'data', 'config']
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Missing directory: {dir_name}"
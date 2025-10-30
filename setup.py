from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_file = this_directory / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="reit-sentiment-analysis",
    version="0.1.0",
    description="REIT Sentiment Analysis Using Employee Reviews",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Konain Niaz",
    author_email="kn4792@g.rit.edu",
    url="https://github.com/kn4792/reit-sentiment-analysis",
    
    # Automatically find all packages in the project
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    python_requires=">=3.8",
    
    # Core dependencies (minimal for now)
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.20.0",
    ],
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "pylint>=3.0.0",
        ],
    },
    
    # Include package data
    include_package_data=True,
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
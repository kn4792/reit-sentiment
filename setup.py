"""
Setup script for REIT Sentiment Analysis package.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="reit-sentiment-analysis",
    version="1.0.0",
    author="Konain Niaz",
    author_email="kn4792@rit.edu",
    description="Employee Sentiment Analysis for REIT Performance Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kn4792/reit-sentiment",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "pylint>=2.17.5",
        ],
        "gpu": [
            "torch>=2.0.1",  # With CUDA support
        ],
    },
    entry_points={
        "console_scripts": [
            "reit-scrape=scripts.scrape_reviews:main",
            "reit-clean=scripts.clean_data:main",
            "reit-sentiment=scripts.sentiment_analysis:main",
            "reit-keywords=scripts.keyword_extraction:main",
            "reit-test=scripts.run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.json", "config/*.sql"],
    },
    zip_safe=False,
)
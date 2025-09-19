"""
Setup script for HER2+ Breast Cancer Classification Pipeline

This setup.py file allows for proper package installation and distribution.

Authors:
    - Primary: T. Buathongtanakarn
    - AI Assistant: GitHub Copilot

Version: 2.1.0
Last Updated: September 17, 2025
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md for long description"""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "HER2+ Breast Cancer Classification Pipeline"

# Read requirements
def read_requirements():
    """Read requirements.txt for dependencies"""
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="her2-cancer-pipeline",
    version="2.1.0",
    author="T. Buathongtanakarn",
    author_email="scientan@gmail.com",  # Update with actual email
    description="Deep Learning Pipeline for HER2+ Breast Cancer Classification in Whole Slide Images",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thanakornbua/pathothai",  # Update with actual repo
    
    packages=find_packages(),
    include_package_data=True,
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "her2-pipeline=her2_pipeline:main",
            "her2-validate=validate_setup:main",
        ],
    },
    
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/yourusername/her2-cancer-pipeline/issues",
        "Source": "https://github.com/yourusername/her2-cancer-pipeline",
        "Documentation": "https://github.com/yourusername/her2-cancer-pipeline/wiki",
    },
    
    keywords=[
        "deep-learning",
        "medical-imaging",
        "pathology",
        "breast-cancer",
        "her2",
        "whole-slide-imaging",
        "pytorch",
        "monai",
        "computer-vision",
        "healthcare-ai",
    ],
)

"""
Setup script for Clinical Topic Modeling Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="clinical-topic-modeling",
    version="1.0.0",
    author="Clinical AI Research Team",
    author_email="contact@clinical-ai.com",
    description="A comprehensive framework for clinical topic modeling and prediction using BERT-based approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clinical-topic-modeling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
        ],
    },
    entry_points={
        "console_scripts": [
            "clinical-topic-modeling=framework:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "clinical",
        "medical",
        "nlp",
        "bert",
        "topic-modeling",
        "machine-learning",
        "healthcare",
        "prediction",
        "classification",
        "deep-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/clinical-topic-modeling/issues",
        "Source": "https://github.com/yourusername/clinical-topic-modeling",
        "Documentation": "https://yourusername.github.io/clinical-topic-modeling/",
    },
)
#!/usr/bin/env python3
"""
Setup script for ΨQRH Transformer

This file provides backward compatibility for tools that don't support pyproject.toml.
The main configuration is in pyproject.toml.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="psiqrh",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    author="Klenio Araujo Padilha",
    author_email="klenioaraujo@gmail.com",
    description="ΨQRH Transformer: Quaternionic-Harmonic Architecture for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs",
    project_urls={
        "Bug Tracker": "https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs/issues",
        "Documentation": "https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs/docs",
        "Source Code": "https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs",
        "DOI": "https://zenodo.org/records/17171112",
    },
    packages=find_packages(exclude=["tests", "docs", "examples.reuse_guides"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pylint>=2.17.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "psiqrh=psiqrh:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
)
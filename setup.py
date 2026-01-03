"""Setup script for speculative-cascade package."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="speculative-cascade",
    version="0.1.0",
    author="Marco Durán Cabobianco",
    author_email="marco@anachroni.co",
    description="Cascading Speculative Acceleration for LLM Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anacronic-io/speculative-cascade",
    project_urls={
        "Bug Tracker": "https://github.com/anacronic-io/speculative-cascade/issues",
        "Documentation": "https://github.com/anacronic-io/speculative-cascade#readme",
        "Source Code": "https://github.com/anacronic-io/speculative-cascade",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cascade-bench=speculative_cascade.benchmarks.cli:main",
            "cascade-infer=speculative_cascade.cli:main",
        ],
    },
)

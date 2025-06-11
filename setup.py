#!/usr/bin/env python3
# setup.py

import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read_requirements(fname="requirements.txt"):
    path = os.path.join(here, fname)
    with io.open(path, encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

# long_description from your README if you have one
def read_long_description(fname="README.md"):
    path = os.path.join(here, fname)
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="neuroprin",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="NeuroPRIN: Pruned Resonance Inference Network for time-series trading",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuroprin",
    packages=find_packages(exclude=["examples*", "scripts*", "tests*"]),
    install_requires=read_requirements(),
    python_requires=">=3.7",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "neuroprin-demo=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

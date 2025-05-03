#!/usr/bin/env python
# Created by "Thieu" at 16:08, 03/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
    name="probnet",
    version="0.1.0",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="ProbNet: A Unified Probabilistic Neural Network Framework for Classification and Regression Tasks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=[
        "Probabilistic Neural Network", "PNN", "General Regression Neural Network",
        "GRNN", "Regression", "Supervised Learning", "Gaussian function",
        "Kernel-based Models", "Gaussian Kernel", "Non-parametric Learning",
        "multi-input multi-output (MIMO)",
        "hybrid learning", "vectorized implementation", "interpretable learning", "explainable AI (XAI)",
        "Python Library", "Machine Learning Framework", "Model Deployment", "Neural Network API",
        "machine learning", "regression", "classification", "time series forecasting",
        "soft computing", "computational intelligence", "intelligent systems",
        "Scikit-learn Compatible", "Lightweight ML Library", "Extendable Neural Networks",
        "Interpretable Machine Learning", "Plug-and-Play ML Models", "Fast Model Prototyping",
        "intelligent decision system", "adaptive system", "simulation studies"
    ],
    url="https://github.com/thieu1995/ProbNet",
    project_urls={
        'Documentation': 'https://probnet.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/ProbNet',
        'Bug Tracker': 'https://github.com/thieu1995/ProbNet/issues',
        'Change Log': 'https://github.com/thieu1995/ProbNet/blob/main/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=1.17.1", "scipy>=1.7.1", "scikit-learn>=1.2.1",
                      "pandas>=1.3.5", "permetrics>=2.0.0"],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov==4.0.0", "flake8>=4.0.1"],
    },
    python_requires='>=3.8',
)

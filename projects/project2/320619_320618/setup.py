from setuptools import setup, find_packages

setup(
    name="AUTOML-PACKAGE",  # Replace with your package's name
    version="0.1",
    description="A Python package for automated data preprocessing and evaluation pipeline.",
    author="Maciej and Michal",
    url="https://github.com/BorkowskiMaciej/automl-package",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

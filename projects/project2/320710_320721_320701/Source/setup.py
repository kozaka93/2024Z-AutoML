from setuptools import setup, find_packages

setup(
    name="automlclassifier",
    version="0.1.0",
    description="Automation of machine learning with visualizations and report generation.",
    author="Jan Kruszewski, Bartosz Maj, Bartosz Olszewski",
    author_email="bartek9568@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "numpy>=1.21",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "xgboost>=1.5",
    ],
    python_requires=">=3.6",
)

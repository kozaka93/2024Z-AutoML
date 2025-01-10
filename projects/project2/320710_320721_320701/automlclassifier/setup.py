from setuptools import setup, find_packages
setup(
    name="automlclassifier",
    version="0.1.0",
    description="Automatyzacja uczenia maszynowego z wizualizacjami i generowaniem raportów.",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "numpy>=1.21",
        "matplotlib>=3.4",
        "seaborn>=0.11",
    ],
)

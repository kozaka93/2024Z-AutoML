from setuptools import setup, find_packages
from codecs import open


with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

with open("requirements.txt", 'r', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setup(
    name="medaid",
    version="0.1.7",
    description="Automated Machine Learning for medical use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeptuchMateusz/medAId",
    author="Zofia Kamińska, Karolina Dunal, Mateusz Deptuch",
    license="MIT",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    install_requires=install_requires,
    include_package_data=True,
    python_requires='>=3.8',
    keywords=[
        "automated machine learning",
        "automl",
        "machine learning",
        "medical data",
    ],
    test_suite="tests",
    tests_require=["pytest", "unittest2"],
)

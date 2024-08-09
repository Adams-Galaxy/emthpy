from setuptools import setup, find_packages

setup(
    name="emthpy",
    version="0.1.0",
    description="A sample Python package",
    author="Adam Williams",
    author_email="-",
    packages=find_packages(),  # Automatically finds subpackages
    install_requires=[  # Dependencies
        "numpy",
    ],
)

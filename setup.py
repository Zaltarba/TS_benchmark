from setuptools import setup, find_packages

setup(
    name="compactformer",
    version="0.1.0",
    description="Compact Transformer Variants for Time Series Forecasting",
    author="Ali Forootan",
    author_email="aliforootan@ieee.org",
    url="https://github.com/yourusername/compactformer",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.4",
        "pandas==1.5.3",
        "matplotlib==3.7.1",
        "scikit-learn==1.2.2",
        "scipy==1.10.1",
        "torch==2.0.1"
    ],
    python_requires=">=3.10.6",
    include_package_data=True,
    package_data={
        # "compactformer": ["simulation_results/*.csv"],
    },
)

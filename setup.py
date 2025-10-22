from setuptools import setup, find_packages

setup(
    name="risksim",
    version="0.1.0",
    author="Thomas R. Holy",
    author_email="thomas.robert.holy@gmail.com",
    description="RiskSim! Simulate financial risk!",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/trholy/risksim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    license_files=('LICENSE',),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.10.7",
        "mkl>=2024.2.2",
        "numpy>=2.2.6",
        "pandas>=2.3.3",
        "scikit-learn>=1.7.2",
        "streamlit>=1.15.2"
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff"
        ]
    },
    test_suite='pytest'
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="video-games-ml",
    version="1.0.0",
    author="Pablo-HF-ES",
    description="Machine Learning model for predicting video game launch price degradation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pablo-HF-ES/An-lisis-de-la-disminuici-n-del-precio-de-salida-de-viodeojuegos",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas==3.0.3",
        "scikit-learn==1.9.1",
        "joblib==1.5.3",
        "flask==3.0.0",
        "flask-cors==4.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vg-ml-train=src.train:main",
            "vg-ml-predict=src.predict:main",
            "vg-ml-api=app:main",
        ],
    },
)

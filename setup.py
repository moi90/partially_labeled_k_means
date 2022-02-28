from setuptools import setup

import versioneer

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="constrained_kmeans",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simon-Martin Schroeder",
    author_email="martin.schroeder@nerdluecht.de",
    # description="Automates machine learning and other computer experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/moi90/constrained_kmeans",
    packages=["constrained_kmeans"],
    include_package_data=True,
    install_requires=["scikit-learn", "scipy", "numpy", "joblib"],
    python_requires=">=3.6",
    extras_require={
        "tests": ["pytest", "pytest-cov", "pytest-benchmark"],
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinxcontrib-programoutput",
            "sphinx-autodoc-typehints>=1.10.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)

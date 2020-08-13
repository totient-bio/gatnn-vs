from setuptools import setup, find_packages
setup(
    name="GAtNN-VS",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.4", "dgl>=0.4.0", "rdkit", 
        "tqdm", "hydra-core", "omegaconf", "pandas", "cytoolz", "feather-format"],

    # metadata to display on PyPI
    author="Totient inc.",
    author_email="me@example.com",
    description="Neural network based virtual screening toolkit",
    keywords="graph,attention,neural network,chemistry,virtual screen",
    url="https://github.com/totient-bio/gatnn-vs",
    project_urls={
        "Bug Tracker": "https://github.com/totient-bio/gatnn-vs/issues",
        "Documentation": "https://github.com/totient-bio/gatnn-vs",
        "Source Code": "https://github.com/totient-bio/gatnn-vs",
    },
    data_files = [("config", ["train-config.yaml", "eval-config.yaml"])],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    python_requires=">=3.6"
)
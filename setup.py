from setuptools import setup, find_packages
setup(
    name="gatnnvs",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.4", "dgl>=0.4.0",  # "rdkit", - there is no pip package
        "tqdm", "hydra-core", "pandas", "cytoolz", "feather-format", "tensorboard"],

    # metadata to display on PyPI
    author="Totient inc.",
    author_email="opensource@totient.bio",
    description="Neural network based virtual screening toolkit",
    keywords="graph,attention,neural network,chemistry,virtual screen",
    url="https://github.com/totient-bio/gatnn-vs",
    project_urls={
        "Bug Tracker": "https://github.com/totient-bio/gatnn-vs/issues",
        "Documentation": "https://github.com/totient-bio/gatnn-vs",
        "Source Code": "https://github.com/totient-bio/gatnn-vs",
    },
    include_package_data=True,
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
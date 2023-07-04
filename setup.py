import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sophia",
    version="0.0.1",
    author="Hong Liu",
    author_email="hliu99@stanford.edu",
    description="Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Liuhong99/Sophia",
    py_modules = ["sophia"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LION",
    version="0.1",
    author="Emily Costa",
    author_email="costa.em@northeastern.edu",
    description="A comprehensive library for analyzing Darshan logs collected on high-performance computing (HPC) systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GoodwillComputingLabs/LION/",
    project_urls={
        "Bug Tracker": "https://github.com/GoodwillComputingLabs/LION/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
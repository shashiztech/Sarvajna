from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sarvanjna",
    version="0.1.0",
    author="Research AI Team",
    description="A production-grade multimodal AI platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="sarvanjna"),
    package_dir={"": "sarvanjna"},
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

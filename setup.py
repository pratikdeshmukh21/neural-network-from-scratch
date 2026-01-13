from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simple-nn-mnist",
    version="1.0.0",
    author="Pratik Deshmukh",
    author_email="pratikdeshmukh212121@gmail.com",  # you can change later
    description="Neural Network from scratch using NumPy (MNIST)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
)

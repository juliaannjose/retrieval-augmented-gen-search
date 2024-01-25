from setuptools import setup, find_packages

setup(
    name="retrieval-augmented-gen-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai==1.9.0",
        "pymilvus",
    ],
)

from setuptools import setup, find_packages

setup(
    name="retrieval-augmented-gen-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai==1.9.0",
        "pymilvus==2.3.6",
        "streamlit==1.30.0",
        "psycopg2-binary==2.9.9",
    ],
)

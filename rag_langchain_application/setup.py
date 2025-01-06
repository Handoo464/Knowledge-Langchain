from setuptools import setup, find_namespace_packages

setup(
    name="rag_langchain_application",
    version="0.1",
    packages=find_namespace_packages(include=["src*"]),
    install_requires=[
        "fastapi",
        "uvicorn",
        "langchain",
        "langchain-community",
        "transformers",
        "torch",
        "pydantic"
    ]
) 
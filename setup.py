from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-PROJECT-10",
    version="0.1",
    author="Himanshu Borikar",
    packages=find_packages(),
    install_requires=requirements
) 
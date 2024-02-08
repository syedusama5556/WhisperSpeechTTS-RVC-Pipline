import os

from setuptools import find_packages, setup


def get_requirements(path):
    with open(path, encoding="utf-8") as requirements:
        return [requirement.strip() for requirement in requirements]


base_dir = os.path.dirname(os.path.abspath(__file__))
install_requires = get_requirements(os.path.join(base_dir, "requirements.txt"))

try:
    long_description = open("README.md").read()
except IOError:
    long_description = ""

setup(
    name="dlas",
    version="0.0.1",
    description="",
    author="James Betker",
    author_email="james@adamant.ai",
    packages=find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    classifiers=[],
    python_requires=">=3.8",
)

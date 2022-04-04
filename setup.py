from setuptools import setup, find_packages

setup(
      name='sapai_gym',
      version='0.1',
      packages=find_packages(),
      install_requires=[
          "sapai @ git+https://github.com/manny405/sapai.git@main",
          "gym~=0.21.0",
          "sklearn"
      ]
)

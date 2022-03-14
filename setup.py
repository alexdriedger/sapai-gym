from distutils.core import setup

setup(
      name='sapai-gym',
      version='0.1',
      packages=['sapai-gym'],
      install_requires=[
          # 'git+https://github.com/manny405/sapai.git',
          "gym~=0.22.0"
          "numpy~=1.22.2"
          "sklearn"
      ],
      data_files=[],
)

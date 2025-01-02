from setuptools import setup
from setuptools import find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()
    
# requires = parse_requirements('requirements.txt')

setup(name='common_utils',
      version='0.0.1',
      description='Simple scripts for common tasks.',
      author='The fastest man alive.',
      packages=find_packages(),
      install_requires=[])
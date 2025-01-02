from setuptools import setup
from setuptools import find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()
    
# requires = parse_requirements('requirements.txt')

setup(name='melody',
      version='0.0.2',
      description='fucking audio processing',
      author='The fastest man alive.',
      packages=find_packages(),
      install_requires=[])
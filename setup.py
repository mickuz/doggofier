from setuptools import setup, find_packages

setup(name='Doggofier',
      version='0.2',
      description='The image classifier for dog breeds',
      author='Michal Kuzniewicz',
      author_email='michal.kuzniewicz@tuta.io',
      url='https://github.com/mickuz/doggofier',
      packages=find_packages(include=['doggofier', 'doggofier.*']),
      install_requires=[
            'numpy',
            'torch',
            'torchvision'
      ],
      extras_require={
            'plotting': ['matplotlib', 'jupyter']
      },
      setup_requires=[
            'flake8'
      ])

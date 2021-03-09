from setuptools import setup, find_packages

setup(name='Doggofier',
      version='0.4',
      description='The image classifier for dog breeds',
      author='Michal Kuzniewicz',
      author_email='michal.kuzniewicz@tuta.io',
      url='https://github.com/mickuz/doggofier',
      packages=find_packages(include=['doggofier', 'doggofier.*']),
      install_requires=[
            'numpy',
            'pandas',
            'torch',
            'torchvision'
      ],
      extras_require={
            'plotting': ['matplotlib', 'jupyter'],
            'models': ['torchsummary']
      },
      setup_requires=[
            'flake8'
      ])

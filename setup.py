from setuptools import setup, find_packages

setup(name='Doggofier',
      version='1.0',
      description='The image classifier for dog breeds',
      author='Michal Kuzniewicz',
      author_email='michal.kuzniewicz@tuta.io',
      url='https://github.com/mickuz/doggofier',
      packages=find_packages(include=['doggofier', 'doggofier.*']),
      install_requires=[
            'numpy',
            'pandas',
            'torch',
            'torchvision',
            'flask',
            'gunicorn'
      ],
      extras_require={
            'plotting': ['matplotlib', 'jupyter'],
            'models': ['torchsummary']
      },
      setup_requires=[
            'flake8'
      ],
      scripts=[
            'doggofier/train.py',
            'doggofier/evaluate.py'
      ])

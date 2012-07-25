try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Salienpy',
    'author': 'Rafael Lopes',
    'url': '.',
    'download_url': 'Where to download it.',
    'author_email': 'rafaellg@vision.ime.usp.br',
    'version': '0.1',
    'install_requires': ['nose', 'opencv'],
    'packages': ['salienpy'],
    'scripts': [],
    'name': 'salienpy'
}

setup(**config)

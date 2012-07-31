try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = "salienpy",
    version = "0.0.1",
    author = "Rafael Lopes",
    author_email = "rafaellg@vision.com",
    description = ("A image visual saliency toolbox"),
    license = "BSD",
    keywords = "saliency computer_vision image",
    url = "http://packages.python.org/salienpy",
    packages=['salienpy', 'tests'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: BSD License",
    ],
)

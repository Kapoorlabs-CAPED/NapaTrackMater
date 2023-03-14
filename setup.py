
import setuptools
from setuptools import find_packages, setup
import os

_dir = os.path.dirname(__file__)

with open('README.md') as f:
    long_description = f.read()

with open(os.path.join(_dir,'napatrackmater','version.py'), encoding="utf-8") as f:
    exec(f.read())
setup(
    name="napatrackmater",

    version=__version__,

    author='Varun Kapoor, Mari Tolonen',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/kapoorlab/NapaTrackMater/',
    description='Import Trackmate XML files for Track Visualization and analysis in Napari.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "lxml",
        "vollseg",
        "napari",
        "natsort",
        "seaborn",
    ],
    entry_points = {
        'console_scripts': [
            'track = napatrackmater.__main__:main',
        ]
    },
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)


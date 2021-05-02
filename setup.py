from setuptools import setup
from setuptools import find_packages
import setuptools
with open('README.md') as f:
    long_description = f.read()


setup(name="napatrackmater",
      version='2.1.1',
      author='Varun Kapoor,Claudia Carabana Garcia',
      author_email='randomaccessiblekapoor@gmail.com',
      url='https://github.com/kapoorlab/NapaTrackMater/',
      description='Import Trackmate XML files for Track Visualization and analysis in Napari.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=["numpy", "pandas", "napari","pyqt5", "natsort", "scikit-image", "scipy", "tifffile", "matplotlib", "ffmpeg-python", "imageio_ffmpeg"],
      packages= ['napatrackmater','napatrackmater/napari_animation','napatrackmater/napari_animation/_qt' ],
      classifiers=['Development Status :: 3 - Alpha',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.7',
                   ])

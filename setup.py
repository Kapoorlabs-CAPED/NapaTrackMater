from setuptools import setup
from setuptools import find_packages

with open('README.md') as f:
    long_description = f.read()


setup(name="napatrackmater",
      version='1.0.0',
      author='Varun Kapoor',
      author_email='randomaccessiblekapoor@gmail.com',
      url='https://github.com/kapoorlab/NapaTrackMater/',
      description='Import Trackmate XML files for Track Visualization and analysis in Napari.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=["numpy", "pandas", "napari==0.4.3", "pyqt5", "btrack","natsort", "scikit-image", "scipy", "opencv-python-headless", "tifffile", "matplotlib", "ffmpeg-python", "imageio_ffmpeg"],
      packages=find_packages(),
      classifiers=['Development Status :: 3 - Alpha',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.9',
                   ])

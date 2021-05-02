# NapaTrackMater
Napari Visualization tool for Trackmate > 6.0 and bTrackmate XML files for 3D + time tracks.


This repository is the bridge between the Fiji and Napari world for exporting and viewing the track XML files using [Napari track layer](https://napari.org/tutorials/fundamentals/tracks.html)

## Installation
This package can be installed by 

`pip install napatrackmater`

If you are building this from the source, clone the repository and install via

```
git clone https://github.com/kapoorlab/NapaTrackMater/

cd NapaTrackMater

pip install -e .

# or, to install in editable mode AND grab all of the developer tools
# (this is required if you want to contribute code back to NapaTrackMater)
pip install -r requirements.txt
```

[![Build Status](https://travis-ci.com/kapoorlab/napatrackmater.svg?branch=master)](https://travis-ci.com/github/kapoorlab/napatrackmater)
[![PyPI version](https://img.shields.io/pypi/v/napatrackmater.svg?maxAge=2591000)](https://pypi.org/project/napatrackmater/)


## Usage

To use this repository you need to have an XML file coming either from the Fiji plugin Trackmate version>6.0 or from bTrackmate version>2.0.
Both the programs save the same XML file containing the information about your tracking session. This XML file can be used to re-generate trackscheme along with the tracks they came from, for example a typical trackscheme with tracks overlay in Fiji:


![Track Scheme](https://github.com/kapoorlab/NapaTrackMater/blob/main/Images/trackscheme.png)


In this scheme there are some cells that divide multiple times and some don't, we can view these tracks by using the tracks layer of Napari.


In Napari tracks layer view we break the dividing trajectory into components that are called tracklets. These tracklets represent the trajectory of individual child cell or the root cell. 


More than just viewing the tracks we can extract the following special functions from them:


1) If the cells move inside a tissue we can calculate the distance of the cells in a track from the tissue boundary for the root track and the following children tracklets, this gives a cell fate determination plot which shows the starting and the ending distance of each tracklet.
Check the [notebook](https://github.com/kapoorlab/NapaTrackMater/blob/main/examples/CellFateDetermination.ipynb).


2) If the cells had an intensity oscillation we can compute the frequency of such oscillation for each tracklet of the track by Fourier transforming the intensity over time.
Check the [notebook](https://github.com/kapoorlab/NapaTrackMater/blob/main/examples/FrequencyOscillations.ipynb).

## Example
We provide an [example dataset](https://github.com/kapoorlab/NapaTrackMater/tree/main/examples/data) of C. elegans, click on the link provided in the txt file to obtain the tif files of the Raw, Segmentation and the Mask image. Create a local enviornment to run the example by downloading the tif files and putting them in the save directory along with the csv and xml files. We create this csv file using the Raw and Segmentation image using this [notebook](https://github.com/kapoorlab/NapaTrackMater/blob/main/examples/BTrackMateLocalization.ipynb) then we did the tracking via bTrackmate using the csv file to obtain the xml file in the save directory. To view the intensity oscillations we do not use a mask of the tissue surrounding the cells and obtain the intensity oscillation of all cells.

## Requirements

- Python 3.9 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Claudia Carabana Garcia

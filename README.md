# NapaTrackMater
Napari Visualization tool for Trackmate > 6.0 and bTrackmate XML files for 3D + time tracks

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

Import [Trackmate](https://imagej.net/TrackMate) XML files for visualization in Napari using [Tracks Layer](https://napari.org/tutorials/fundamentals/tracks.html) .

## Usage

To use this repository you need to have an XML file coming either from the Fiji plugin Trackmate version>6.0 or from bTrackmate version>2.0.
Both the programs save the same XML file containing the information about your tracking session. This XML file can be used to re-generate trackscheme along with the tracks they came from, for example a typical trackscheme with tracks overlay in Fiji:
![Track Scheme](https://github.com/kapoorlab/NapaTrackMater/blob/main/Images/trackscheme.png)

Check the [notebook](https://github.com/kapoorlab/NapaTrackMater/blob/main/CellFateDetermination.ipynb).

## Requirements

- Python 3.9 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Claudia Carabana Garcia

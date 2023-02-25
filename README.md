# NapaTrackMater
Napari Visualization tool for Trackmate > 6.0 and bTrackmate XML files for 3D + time tracks.

This repository is the bridge between the Fiji and Napari world for exporting and viewing the track XML files using [Napari track layer](https://napari.org/tutorials/fundamentals/tracks.html).

[![Build Status](https://travis-ci.com/kapoorlab/napatrackmater.svg?branch=master)](https://travis-ci.com/github/kapoorlab/napatrackmater)
[![PyPI version](https://img.shields.io/pypi/v/napatrackmater.svg?maxAge=2591000)](https://pypi.org/project/napatrackmater/)

## Installation
This package can be installed with:

`pip install --user napatrackmater`

If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/kapoorlab/NapaTrackMater/

cd NapaTrackMater

pip install --user -e .

# or, to install in editable mode AND grab all of the developer tools
# (this is required if you want to contribute code back to NapaTrackMater)
pip install --user -r requirements.txt
```

### Pipenv install

Pipenv allows you to install dependencies in a virtual environment.

```bash
# install pipenv if you don't already have it installed
pip install --user pipenv

# clone the repository and sync the dependencies
git clone https://github.com/kapoorlab/NapaTrackMater/
cd NapaTrackMater
pipenv sync

# make the current package available
pipenv run python setup.py develop

# you can run the example notebooks by starting the jupyter notebook inside the virtual env
pipenv run jupyter notebook
```


## Docker

A Docker image can be used to run the code in a container. Once inside the project's directory, build the image with:

~~~bash
docker build -t kapoorlab/NapaTrackMater .
~~~

Now to run the `track` command:

~~~bash
# show help
docker run --rm -it kapoorlab/NapaTrackMater
# run it with example data
docker run --rm -it -v $(pwd)/examples/data:/input kapoorlab/NapaTrackMater track -f /input -r /input/Raw.tif -s /input/Seg.tif -s /input/Mask.tif -n test
~~~

## Requirements

- Python 3.9 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Mari Tolonen

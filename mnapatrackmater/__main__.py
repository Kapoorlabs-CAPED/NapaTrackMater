#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Entrypoint for the CreateTrackCheckpoint command
"""
import argparse

from . import bTrackmate

def main():
    """ Parse arguments and launch processor """
    parser = argparse.ArgumentParser(
        description='Create track checkpoint'
    )

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument(
        '-f',
        '--folder',
        metavar='/path/to/output-folder',
        type=str,
        help='path to output folder',
    )

    requiredNamed.add_argument(
        '-r',
        '--raw',
        metavar='/path/to/raw.tif',
        type=str,
        help='path to raw tif',
    )

    requiredNamed.add_argument(
        '-s',
        '--seg',
        metavar='/path/to/seg.tif',
        type=str,
        help='path to segmentation tif',
    )

    requiredNamed.add_argument(
        '-m',
        '--mask',
        metavar='/path/to/mask.tif',
        type=str,
        help='path to mask tif',
    )

    requiredNamed.add_argument(
        '-n',
        '--name',
        metavar='Some name',
        type=str,
        help='A name for identification',
    )

    arguments = parser.parse_args()

    bTrackmate.CreateTrackCheckpoint(arguments.raw, arguments.seg, arguments.mask, arguments.name, arguments.folder)

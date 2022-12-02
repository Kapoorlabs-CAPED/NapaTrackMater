import codecs
import csv
import math
import multiprocessing
import os
import xml.etree.cElementTree as et
from functools import partial
from pathlib import Path
import sys
from multiprocessing import Pool, Process, Queue
from PyQt5.QtCore import pyqtSlot
from tqdm import tqdm
import time as clock
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
from scipy import spatial
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from skimage import measure
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from tifffile import imread, imwrite
from .napari_animation import AnimationWidget
import dask as da
from dask.array.image import imread as daskread
from skimage.util import map_array
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit import Model
from numpy import exp, loadtxt, pi, sqrt
from matplotlib import cm
from qtpy.QtCore import Qt
'''Define function to run multiple processors and pool the results together'''





Boxname = 'TrackBox'
AttributeBoxname = 'AttributeIDBox'
TrackAttributeBoxname = 'TrackAttributeIDBox'
pd.options.display.float_format = '${:,.2f}'.format
savedir = None

ParentDistances = {}
ChildrenDistances = {}
timed_mask = {}
AllStartParent = {}
AllEndParent = {}
AllID = []
AllStartChildren = {}
AllEndChildren = {}
DividingTrackIds = []
NonDividingTrackIds = []
AllTrackIds = []
SaveIds = []

globalcount = "0"
parentstartid = []
parentstartdist = []
parentendid = []
parentenddist = []
childrenstartid = []
childrenstartdist = []
childrenendid = []
childrenenddist = []


def run_multiprocessing(func, track, filtered_track_ids, Uniqueproperties ):
    process_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    


def prob_sigmoid(x):
    return 1 - math.exp(-x)


def CreateTrackCheckpoint(ImageName, LabelName, MaskName, Name, savedir):

    Mask = None

    Label = imread(LabelName)

    Image = imread(ImageName)
    
    if len(Image.shape) == 3:
        NewImage = np.zeros([Image.shape[0], 1, Image.shape[1], Image.shape[2]])
        NewImage[:,0,:,:] = Image
        NewLabel = np.zeros([Image.shape[0], 1, Image.shape[1], Image.shape[2]])
        NewLabel[:,0,:,:] = Label
        
        Label = NewLabel
        Image = NewImage
    if MaskName is not None:
            Mask = imread(MaskName)

    assert Image.shape == Label.shape

    TimeList = []

    XList = []
    YList = []
    ZList = []
    LabelList = []
    PerimeterList = []
    VolumeList = []
    IntensityList = []
    ExtentXList = []
    ExtentYList = []
    ExtentZList = []

    print('Image has shape:', Image.shape)
    print('Image Dimensions:', len(Image.shape))

    if Mask is not None:
        if len(Mask.shape) < len(Image.shape):
            # T Z Y X
            UpdateMask = np.zeros(
                [Label.shape[0], Label.shape[1], Label.shape[2], Label.shape[3]]
            )
            for i in range(0, UpdateMask.shape[0]):
                for j in range(0, UpdateMask.shape[1]):

                    UpdateMask[i, j, :, :] = Mask[i, :, :]
            Mask = UpdateMask
    for i in tqdm(range(0, Image.shape[0])):

        CurrentSegimage = Label[i, :].astype('uint16')
        Currentimage = Image[i, :]
        if Mask is not None:
            CurrentSegimage[Mask[i, :] == 0] = 0
        properties = measure.regionprops(CurrentSegimage, Currentimage)
        for prop in properties:

            Z = prop.centroid[0]
            Y = prop.centroid[1]
            X = prop.centroid[2]
            regionlabel = prop.label
            intensity = np.sum(prop.image)
            sizeZ = abs(prop.bbox[0] - prop.bbox[3])
            sizeY = abs(prop.bbox[1] - prop.bbox[4])
            sizeX = abs(prop.bbox[2] - prop.bbox[5])
            volume = sizeZ * sizeX * sizeY
            radius = math.pow(3 * volume / (4 * math.pi), 1.0 / 3.0)
            perimeter = 2 * math.pi * radius
            TimeList.append(i)
            XList.append(int(X))
            YList.append(int(Y))
            ZList.append(int(Z))
            LabelList.append(regionlabel)
            VolumeList.append(volume)
            PerimeterList.append(perimeter)
            IntensityList.append(intensity)
            ExtentZList.append(sizeZ)
            ExtentXList.append(sizeX)
            ExtentYList.append(sizeY)

    df = pd.DataFrame(
        list(
            zip(
                TimeList,
                XList,
                YList,
                ZList,
                LabelList,
                PerimeterList,
                VolumeList,
                IntensityList,
                ExtentXList,
                ExtentYList,
                ExtentZList,
            )
        ),
        index=None,
        columns=[
            'T',
            'X',
            'Y',
            'Z',
            'Label',
            'Perimeter',
            'Volume',
            'Intensity',
            'ExtentX',
            'ExtentY',
            'ExtentZ',
        ],
    )

    df.to_csv(savedir + '/' + 'FijibTMcheckpoint' + Name + '.csv', index=False)



def CreateTrackMate_CSV( LabelName,   savedir):


    Label = imread(LabelName)
    Name = os.path.basename(os.path.splitext(LabelName)[0])


    TimeList = []

    XList = []
    YList = []
    TrackIDList = []
    QualityList = []
    print('Image has shape:', Label.shape)
    print('Image Dimensions:', len(Label.shape))

    CurrentSegimage = Label.astype('uint16')
    properties = measure.regionprops(CurrentSegimage)
    for prop in properties:

            T = prop.centroid[0]
            Y = prop.centroid[1]
            X = prop.centroid[2]
            regionlabel = prop.label
            sizeZ = abs(prop.bbox[0] - prop.bbox[3])
            sizeY = abs(prop.bbox[1] - prop.bbox[4])
            sizeX = abs(prop.bbox[2] - prop.bbox[5])
            volume = sizeZ * sizeX * sizeY
            radius = math.pow(3 * volume / (4 * math.pi), 1.0 / 3.0)
            perimeter = 2 * math.pi * radius
            TimeList.append(int(T))
            XList.append(int(X))
            YList.append(int(Y))
            TrackIDList.append(regionlabel)
            QualityList.append(radius)

    df = pd.DataFrame(
        list(
            zip(
                XList,
                YList,
                TimeList,
                TrackIDList,
                QualityList
            )
        ),
        index=None,
        columns=[
            'POSITION_X',
            'POSITION_Y',
            'FRAME',
            'TRACK_ID',
            'QUALITY'
        ],
    )

    df.to_csv(savedir + '/' + 'TrackMate_csv' + Name + '.csv', index=False)




def GetBorderMask(Mask):

    ndim = len(Mask.shape)
    # YX shaped object
    if ndim == 2:
        Mask = label(Mask)
        Boundary = find_boundaries(Mask)

    # TYX shaped object
    if ndim == 3:

        Boundary = np.zeros([Mask.shape[0], Mask.shape[1], Mask.shape[2]])
        for i in range(0, Mask.shape[0]):

            Mask[i, :] = label(Mask[i, :])
            Boundary[i, :] = find_boundaries(Mask[i, :])

        # TZYX shaped object
    if ndim == 4:

        Boundary = np.zeros(
            [Mask.shape[0], Mask.shape[1], Mask.shape[2], Mask.shape[3]]
        )

        # Loop over time
        for i in range(0, Mask.shape[0]):

            Mask[i, :] = label(Mask[i, :])

            for j in range(0, Mask.shape[1]):

                Boundary[i, j, :, :] = find_boundaries(Mask[i, j, :, :])

    return Boundary


"""
Convert an integer image into boundary points for 2,3 and 4D data

"""


def boundary_points(mask, xcalibration, ycalibration, zcalibration):

    ndim = len(mask.shape)

    # YX shaped object
    if ndim == 2:
        mask = label(mask)
        labels = []
        size = []
        tree = []
        properties = measure.regionprops(mask, mask)
        for prop in properties:

            labelimage = prop.image
            regionlabel = prop.label
            sizey = abs(prop.bbox[0] - prop.bbox[2]) * xcalibration
            sizex = abs(prop.bbox[1] - prop.bbox[3]) * ycalibration
            volume = sizey * sizex
            radius = math.sqrt(volume / math.pi)
            boundary = find_boundaries(labelimage)
            indices = np.where(boundary > 0)
            real_indices = np.transpose(np.asarray(indices)).copy()
            for j in range(0, len(real_indices)):

                real_indices[j][0] = real_indices[j][0] * xcalibration
                real_indices[j][1] = real_indices[j][1] * ycalibration

            tree.append(spatial.cKDTree(real_indices))

            if regionlabel not in labels:
                labels.append(regionlabel)
                size.append(radius)
        # This object contains list of all the points for all the labels in the Mask image with the label id and volume of each label
        timed_mask[str(0)] = [tree, indices, labels, size]

    # TYX shaped object
    if ndim == 3:

        Boundary = find_boundaries(mask)
        for i in tqdm(range(0, mask.shape[0])):

            mask[i, :] = label(mask[i, :])
            properties = measure.regionprops(mask[i, :], mask[i, :])
            labels = []
            size = []
            tree = []
            for prop in properties:

                labelimage = prop.image
                regionlabel = prop.label
                sizey = abs(prop.bbox[0] - prop.bbox[2]) * ycalibration
                sizex = abs(prop.bbox[1] - prop.bbox[3]) * xcalibration
                volume = sizey * sizex
                radius = math.sqrt(volume / math.pi)
                boundary = find_boundaries(labelimage)
                indices = np.where(boundary > 0)
                real_indices = np.transpose(np.asarray(indices)).copy()
                for j in range(0, len(real_indices)):

                    real_indices[j][0] = real_indices[j][0] * ycalibration
                    real_indices[j][1] = real_indices[j][1] * xcalibration

                tree.append(spatial.cKDTree(real_indices))
                if regionlabel not in labels:
                    labels.append(regionlabel)
                    size.append(radius)

            timed_mask[str(i)] = [tree, indices, labels, size]
            
    # TZYX shaped object
    if ndim == 4:

        Boundary = np.zeros(
            [mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]]
        )
        results = []
        x_ls = range(0, mask.shape[0])
        
        results.append(parallel_map(mask, xcalibration, ycalibration, zcalibration, Boundary, i) for i in tqdm(x_ls))
        da.delayed(results).compute()
        

    return timed_mask, Boundary


def parallel_map(mask, xcalibration, ycalibration, zcalibration, Boundary, i):

    mask[i, :] = label(mask[i, :])
    properties = measure.regionprops(mask[i, :], mask[i, :])
    labels = []
    size = []
    tree = []
    for prop in properties:

        regionlabel = prop.label
        sizez = abs(prop.bbox[0] - prop.bbox[3]) * zcalibration
        sizey = abs(prop.bbox[1] - prop.bbox[4]) * ycalibration
        sizex = abs(prop.bbox[2] - prop.bbox[5]) * xcalibration
        volume = sizex * sizey * sizez
        radius = math.pow(3 * volume / (4 * math.pi), 1.0 / 3.0)
        for j in range(mask.shape[1]):
           Boundary[i,j, :, :] = find_boundaries(mask[i, j, :, :])

        indices = np.where(Boundary[i, :] > 0)
       
        real_indices = np.transpose(np.asarray(indices)).copy()
        for j in range(0, len(real_indices)):

            real_indices[j][0] = real_indices[j][0] * zcalibration
            real_indices[j][1] = real_indices[j][1] * ycalibration
            real_indices[j][2] = real_indices[j][2] * xcalibration

        tree.append(spatial.cKDTree(real_indices))
        if regionlabel not in labels:
            labels.append(regionlabel)
            size.append(radius)

    timed_mask[str(i)] = [tree, indices, labels, size]


def analyze_non_dividing_tracklets(root_leaf, spot_object_source_target):

    non_dividing_tracklets = []
    if len(root_leaf) > 0:
        Root = root_leaf[0]
        Leaf = root_leaf[-1]
        tracklet = []
        trackletspeed = []
        trackletdirection = []
        trackletid = 0
        # For non dividing trajectories iterate from Root to the only Leaf
        while Root != Leaf:
            for (
                source_id,
                target_id,
                edge_time,
                directional_rate_change,
                speed,
            ) in spot_object_source_target:
                if Root == source_id:
                    tracklet.append(source_id)
                    trackletspeed.append(speed)
                    trackletdirection.append(directional_rate_change)
                    Root = target_id
                    if Root == Leaf:
                        break
                else:
                    break
                
        non_dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
    return non_dividing_tracklets


def analyze_dividing_tracklets(root_leaf, split_points, spot_object_source_target):

    dividing_tracklets = []
    # Make tracklets
    Root = root_leaf[0]

    visited = []
    # For the root we need to go forward
    tracklet = []
    trackletspeed = []
    trackletdirection = []
    tracklet.append(Root)
    trackletspeed.append(0)
    trackletdirection.append(0)
    trackletid = 1
    RootCopy = Root
    visited.append(Root)
    while RootCopy not in split_points and RootCopy not in root_leaf[1:]:
            for (
                    source_id,
                    target_id,
                    edge_time,
                    directional_rate_change,
                    speed,
                ) in spot_object_source_target:
                    # Search for the target id corresponding to leaf
                    if RootCopy == source_id:

                        # Once we find the leaf we move a step fwd to its target to find its target
                        RootCopy = target_id
                        if RootCopy in split_points:
                            break
                        if RootCopy in visited:
                            break
                        visited.append(target_id)
                        tracklet.append(source_id)
                        trackletspeed.append(speed)
                        trackletdirection.append(directional_rate_change)
    dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
    trackletid = 2

    # Exclude the split point near root
    for i in range(0, len(split_points )):
        Start = split_points[i]
        tracklet = []
        trackletspeed = []
        trackletdirection = []
        tracklet.append(Start)
        trackletspeed.append(0)
        trackletdirection.append(0)
        Othersplit_points = split_points.copy()
        Othersplit_points.pop(i)
        
        for (
                        source_id,
                        target_id,
                        edge_time,
                        directional_rate_change,
                        speed,
                    ) in spot_object_source_target:
                        
                        if Start in visited:
                           break 
                        if Start in Othersplit_points:
                            break
                        if Start == target_id:

                            Start = source_id
                            
                            tracklet.append(source_id)
                            visited.append(source_id)
                            trackletspeed.append(speed)
                            trackletdirection.append(directional_rate_change)
                            

        dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
        trackletid = trackletid + 1

    for i in range(0, len(root_leaf)):
        leaf = root_leaf[i]
        # For leaf we need to go backward
        tracklet = []
        trackletspeed = []
        trackletdirection = []
        tracklet.append(leaf)
        trackletspeed.append(0)
        trackletdirection.append(0)
        while leaf not in split_points and leaf != Root:
            for (
                    source_id,
                    target_id,
                    edge_time,
                    directional_rate_change,
                    speed,
                ) in spot_object_source_target:
                    # Search for the target id corresponding to leaf
                    if leaf == target_id:
                        # Include the split points here

                        # Once we find the leaf we move a step back to its source to find its source
                        leaf = source_id
                        if leaf in split_points:
                            break
                        if leaf in visited:
                            break
                        if leaf == Root:
                            break
                        visited.append(source_id)
                        tracklet.append(source_id)
                        trackletspeed.append(speed)
                        trackletdirection.append(directional_rate_change)
        dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
        trackletid = trackletid + 1

    return dividing_tracklets


def tracklet_properties(
    alltracklets,
    Uniqueobjects,
    Uniqueproperties,
    Mask,
    Segimage,
    TimedMask,
    calibration,
    DividingTrajectory
):

    location_prop_dist = {}

    for i in range(0, len(alltracklets)):
        current_location_prop_dist = []
        trackletid, tracklets, trackletspeeds, trackletdirections = alltracklets[i]
        location_prop_dist[trackletid] = [trackletid]
        for k in range(0, len(tracklets)):

            tracklet = tracklets[k]
            trackletspeed = trackletspeeds[k]
            trackletdirection = trackletdirections[k]
            cell_source_id = tracklet
            frame, z, y, x = Uniqueobjects[int(cell_source_id)]
         
            total_intensity, mean_intensity, real_time, cellradius, pixel_volume = Uniqueproperties[
                int(cell_source_id)
            ]

            if Mask is not None:

                if len(Mask.shape) == 4:  
                  testlocation = (z, y, x)
                if len(Mask.shape) == 3:  
                  testlocation = (y, x)

                tree, indices, masklabel, masklabelvolume = TimedMask[str(int(float(frame)/ calibration[3]))]
                if len(Mask.shape) == 4:
                        region_label = Mask[
                            int(float(frame) / calibration[3]),
                            int(float(z) / calibration[2]),
                            int(float(y) / calibration[1]),
                            int(float(x) / calibration[0]),
                        ]
                if len(Mask.shape) == 3:
                        region_label = Mask[
                            int(float(frame) / calibration[3]),
                            int(float(y) / calibration[1]),
                            int(float(x) / calibration[0]),
                        ]
                for k in range(0, len(masklabel)):
                    currentlabel = masklabel[k]
                    currentvolume = masklabelvolume[k]
                    currenttree = tree[k]
                    # Get the location and distance to the nearest boundary point
                    distance, location = currenttree.query(testlocation)
                    distance = max(0, distance - float(cellradius))
                    if currentlabel == region_label and region_label > 0:
                        prob_inside = prob_sigmoid(distance)
                    else:

                        prob_inside = 0
            else:
                distance = 0
                prob_inside = 0

            current_location_prop_dist.append(
                [
                    frame,
                    z,
                    y,
                    x,
                    total_intensity,
                    mean_intensity,
                    cellradius,
                    distance,
                    prob_inside,
                    trackletspeed,
                    trackletdirection,
                    DividingTrajectory
                ]
            )

        location_prop_dist[trackletid].append(current_location_prop_dist)

    return location_prop_dist


def import_TM_XML_Relabel(xml_path, Segimage,spot_csv, track_csv, savedir, scale = 255 * 255):
    
    print('Reading Image')
    Name = os.path.basename(os.path.splitext(Segimage)[0])
    Segimage = imread(Segimage)
    
    print('Image dimensions:', Segimage.shape)
    root = et.fromstring(codecs.open(xml_path, 'r', 'utf8').read())

    filtered_track_ids = [
        int(track.get('TRACK_ID'))
        for track in root.find('Model').find('FilteredTracks').findall('TrackID')
    ]

    # Extract the tracks from xml
    tracks = root.find('Model').find('AllTracks')
    settings = root.find('Settings').find('ImageData')

    # Extract the cell objects from xml
    Spotobjects = root.find('Model').find('AllSpots')

    # Make a dictionary of the unique cell objects with their properties
    Uniqueobjects = {}
    Uniqueproperties = {}
    AllKeys = []
    AllTrackKeys = []
    AllValues = []
    AllTrackValues = []
    xcalibration = float(settings.get('pixelwidth'))
    ycalibration = float(settings.get('pixelheight'))
    zcalibration = float(settings.get('voxeldepth'))
    tcalibration = int(float(settings.get('timeinterval')))
    spot_dataset = pd.read_csv(spot_csv, delimiter = ',')[3:]
   
    spot_dataset_index = spot_dataset.index
    spot_dataset.keys()
    
    track_dataset = pd.read_csv(track_csv, delimiter = ',')[3:]
   
    track_dataset_index = track_dataset.index
    track_dataset.keys()

    
    
    for k in spot_dataset.keys():
        try:
          
          if k == 'TRACK_ID':
            Track_id = spot_dataset[k].astype('float')  
            indices = np.where(Track_id==0)
            maxtrack_id = max(Track_id)
            condition_indices = spot_dataset_index[indices]
            Track_id[condition_indices] = maxtrack_id + 1
            AllValues.append(Track_id)   
            
            
          if k == 'POSITION_X':
              LocationX = (spot_dataset['POSITION_X'].astype('float')/xcalibration).astype('int')  
              AllValues.append(LocationX)
              
          if k == 'POSITION_Y':
              LocationY = (spot_dataset['POSITION_Y'].astype('float')/ycalibration).astype('int')   
              AllValues.append(LocationY)
          if k == 'POSITION_Z':
              LocationZ = (spot_dataset['POSITION_Z'].astype('float')/zcalibration).astype('int')   
              AllValues.append(LocationZ)
          if k == 'FRAME':
              LocationT = (spot_dataset['FRAME'].astype('float')).astype('int')  
              AllValues.append(LocationT)    
          elif k!='TRACK_ID' and k!='POSITION_X' and k!='POSITION_Y' and k!='POSITION_Z' and k!='FRAME':  
            AllValues.append(spot_dataset[k].astype('float'))
          
          AllKeys.append(k)  
            
        except:
            pass
    
   
    for k in track_dataset.keys():
          if k == 'TRACK_ID':
                    Track_id = track_dataset[k].astype('float')  
                    indices = np.where(Track_id==0)
                    maxtrack_id = max(Track_id)
                    condition_indices = track_dataset_index[indices]
                    Track_id[condition_indices] = maxtrack_id + 1
                    AllTrackValues.append(Track_id)
                    AllTrackKeys.append(k)
          else:  
                  try:   
                     x =  track_dataset[k].astype('float')
                     minval = min(x)
                     maxval = max(x)
                     
                     if minval > 0 and maxval <= 1:
                         
                        x = normalizeZeroOne(x, scale = scale)
                        
                     AllTrackKeys.append(k)
                     AllTrackValues.append(x)
                  except:
                      
                      pass
   
    
    Viz = VizCorrect(Segimage, Name, savedir,AllKeys,AllTrackKeys, AllValues, AllTrackValues, xcalibration, ycalibration, zcalibration, tcalibration)
    Viz.showNapari()



def import_TM_XML_statplots(xml_path,spot_csv, links_csv, savedir, scale = 255 ):
    
    Name = os.path.basename(os.path.splitext(spot_csv)[0])
    root = et.fromstring(codecs.open(xml_path, 'r', 'utf8').read())

    filtered_track_ids = [
        int(track.get('TRACK_ID'))
        for track in root.find('Model').find('FilteredTracks').findall('TrackID')
    ]

    # Extract the tracks from xml
    tracks = root.find('Model').find('AllTracks')
    settings = root.find('Settings').find('ImageData')

    # Extract the cell objects from xml
    Spotobjects = root.find('Model').find('AllSpots')

    # Make a dictionary of the unique cell objects with their properties
    Uniqueobjects = {}
    Uniqueproperties = {}
    AllKeys = []
    
    AllLinkKeys = []
    AllValues = []
    AllLinkValues = []
    
    xcalibration = float(settings.get('pixelwidth'))
    ycalibration = float(settings.get('pixelheight'))
    zcalibration = float(settings.get('voxeldepth'))
    tcalibration = int(float(settings.get('timeinterval')))
    
    spot_dataset = pd.read_csv(spot_csv, delimiter = ',')[3:]
   
    spot_dataset_index = spot_dataset.index
    spot_dataset.keys()
    
    links_dataset = pd.read_csv(links_csv, delimiter = ',')[3:]
   
    links_dataset_index = links_dataset.index
    links_dataset.keys()
    
    for k in spot_dataset.keys():
        try:
          
          if k == 'TRACK_ID':
            Track_id = spot_dataset[k].astype('float')  
            indices = np.where(Track_id==0)
            maxtrack_id = max(Track_id)
            condition_indices = spot_dataset_index[indices]
            Track_id[condition_indices] = maxtrack_id + 1
            AllValues.append(Track_id)
            
            
            
          else:  
            AllValues.append(spot_dataset[k].astype('float'))
          
          AllKeys.append(k)  
            
        except:
            pass
    
    for k in links_dataset.keys():
        
          if k == 'TRACK_ID':
                    Track_id = links_dataset[k].astype('float')  
                    indices = np.where(Track_id==0)
                    maxtrack_id = max(Track_id)
                    condition_indices = links_dataset_index[indices]
                    Track_id[condition_indices] = maxtrack_id + 1
                    AllLinkValues.append(Track_id)
                    AllLinkKeys.append(k)
          else:  
                  try:   
                     x =  links_dataset[k].astype('float')
                  
                        
                     AllLinkKeys.append(k)
                     AllLinkValues.append(x)
                  except:
                      
                      pass
   
    
    Viz = VizCorrect(None, Name, savedir,AllKeys,AllLinkKeys, AllValues, AllLinkValues, xcalibration, ycalibration, zcalibration, tcalibration)
    Viz.showLR()
    


def import_TM_XML_distplots(xml_path, spot_csv, track_csv, savedir, scale = 255):
    
    Name = os.path.basename(os.path.splitext(spot_csv)[0])
    root = et.fromstring(codecs.open(xml_path, 'r', 'utf8').read())

    filtered_track_ids = [
        int(track.get('TRACK_ID'))
        for track in root.find('Model').find('FilteredTracks').findall('TrackID')
    ]

    # Extract the tracks from xml
    tracks = root.find('Model').find('AllTracks')
    settings = root.find('Settings').find('ImageData')

    # Extract the cell objects from xml
    Spotobjects = root.find('Model').find('AllSpots')

    # Make a dictionary of the unique cell objects with their properties
    Uniqueobjects = {}
    Uniqueproperties = {}
    AllKeys = []
    AllTrackKeys = []
    AllValues = []
    AllTrackValues = []
    xcalibration = float(settings.get('pixelwidth'))
    ycalibration = float(settings.get('pixelheight'))
    zcalibration = float(settings.get('voxeldepth'))
    tcalibration = int(float(settings.get('timeinterval')))
    
    spot_dataset = pd.read_csv(spot_csv, delimiter = ',')[3:]
   
    spot_dataset_index = spot_dataset.index
    spot_dataset.keys()
    
    track_dataset = pd.read_csv(track_csv, delimiter = ',')[3:]
   
    track_dataset_index = track_dataset.index
    track_dataset.keys()

    
    
    for k in spot_dataset.keys():
        try:
          
          if k == 'TRACK_ID':
            Track_id = spot_dataset[k].astype('float')  
            indices = np.where(Track_id==0)
            maxtrack_id = max(Track_id)
            condition_indices = spot_dataset_index[indices]
            Track_id[condition_indices] = maxtrack_id + 1
            AllValues.append(Track_id)
            
            
          if k == 'POSITION_X':
              LocationX = (spot_dataset['POSITION_X'].astype('float')/xcalibration).astype('int')  
              AllValues.append(LocationX)
              
          if k == 'POSITION_Y':
              LocationY = (spot_dataset['POSITION_Y'].astype('float')/ycalibration).astype('int')   
              AllValues.append(LocationY)
          if k == 'POSITION_Z':
              LocationZ = (spot_dataset['POSITION_Z'].astype('float')/zcalibration).astype('int')   
              AllValues.append(LocationZ)
          if k == 'FRAME':
              LocationT = (spot_dataset['FRAME'].astype('float')).astype('int')  
              AllValues.append(LocationT)    
          elif k!='TRACK_ID' and k!='POSITION_X' and k!='POSITION_Y' and k!='POSITION_Z' and k!='FRAME':  
            AllValues.append(spot_dataset[k].astype('float'))
          
          AllKeys.append(k)  
            
        except:
            pass
    
   
    for k in track_dataset.keys():
        
          if k == 'TRACK_ID':
                    Track_id = track_dataset[k].astype('float')  
                    indices = np.where(Track_id==0)
                    maxtrack_id = max(Track_id)
                    condition_indices = track_dataset_index[indices]
                    Track_id[condition_indices] = maxtrack_id + 1
                    AllTrackValues.append(Track_id)
                    AllTrackKeys.append(k)
          else:  
                  try:   
                     x =  track_dataset[k].astype('float')
                     minval = min(x)
                     maxval = max(x)
                     
                     if minval > 0 and maxval <= 1:
                         
                        x = normalizeZeroOne(x, scale = scale)
                        
                     AllTrackKeys.append(k)
                     AllTrackValues.append(x)
                  except:
                      
                      pass
   
    
    Viz = VizCorrect(None, Name, savedir,AllKeys,AllTrackKeys, AllValues, AllTrackValues, xcalibration, ycalibration, zcalibration, tcalibration)
    Viz.showWR()

def normalizeZeroOne(x, scale = 255 * 255):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x * scale


class VizCorrect(object):

        def __init__(self, Segimage, Name, savedir,AllKeys, AllTrackKeys, AllValues, AllTrackValues, xcalibration, ycalibration, zcalibration, tcalibration ):
            
            
               self.Segimage = Segimage
               self.Name = Name
               self.savedir = savedir
               self.AllKeys = AllKeys
               self.AllTrackKeys = AllTrackKeys
               self.AllValues = AllValues
               self.AllTrackValues = AllTrackValues
               self.xcalibration = xcalibration
               self.ycalibration = ycalibration
               self.zcalibration = zcalibration
               self.tcalibration = tcalibration
               Path(self.savedir).mkdir(exist_ok=True)
               
               
        def showWR(self):
               
                  
               for k in range(len(self.AllTrackKeys)):
                            if self.AllTrackKeys[k] == 'TRACK_ID':
                                   p = k
                               
                                
               for k in range(len(self.AllTrackKeys)):
                            
                            self.AllTrackID = []
                            self.AllTrackAttr = []                
                            for attr, trackid in tqdm(zip(self.AllTrackValues[k], self.AllTrackValues[p]), total = len(self.AllTrackValues[k])):
                                    
                                            
                                               self.AllTrackID.append(int(float(trackid)))
                                               self.AllTrackAttr.append(float(attr))
                            print('Histplot for: ', self.AllTrackKeys[k])               
                            sns.histplot(self.AllTrackAttr, kde = True)
                            plt.show()
                         
                            
        def showLR(self):
               
                  
               Attr = {}
               
               for k in range(len(self.AllKeys)):
                            if self.AllKeys[k] == 'TRACK_ID':
                                   trackid_key = k
                            if self.AllKeys[k] == 'ID':
                                   spotid_key = k
                            if self.AllKeys[k] == 'FRAME':
                                   frameid_key = k       
                            if self.AllKeys[k] == 'POSITION_Z':
                                   zposid_key = k 
                            if self.AllKeys[k] == 'POSITION_Y':
                                   yposid_key = k
                            if self.AllKeys[k] == 'POSITION_X':
                                   xposid_key = k       
            
               starttime = int(min(self.AllValues[frameid_key]))
               endtime = int(max(self.AllValues[frameid_key]))
                
               for k in range(len(self.AllTrackKeys)):
                      if self.AllTrackKeys[k] == 'SPOT_SOURCE_ID':
                                   sourceid_key = k
                      if self.AllTrackKeys[k] == 'DIRECTIONAL_CHANGE_RATE':
                                   dcr_key = k
                      if self.AllTrackKeys[k] == 'SPEED':
                                   speed_key = k      
                      if self.AllTrackKeys[k] == 'DISPLACEMENT':
                                   disp_key = k              
               
                
               for sourceid, dcrid, speedid, dispid, zposid, yposid, xposid in zip(self.AllTrackValues[sourceid_key], self.AllTrackValues[dcr_key], self.AllTrackValues[speed_key],self.AllTrackValues[disp_key],self.AllValues[zposid_key],self.AllValues[yposid_key],self.AllValues[xposid_key] ):
                   
                   Attr[int(sourceid)] = [dcrid, speedid, dispid, zposid, yposid, xposid]
               
                
               Timedcr = []
               Timespeed = []
               Timedisppos = []
               Timedispneg = []
               
               Timedispposy = []
               Timedispnegy = []
               
               Timedispposx = []
               Timedispnegx = []
               
               
               Alldcrmean = []
               Allspeedmean = []
               Alldispmeanpos = []
               Alldispmeanneg = []
               
               Alldispmeanposx = []
               Alldispmeanposy = []
               
               Alldispmeannegx = []
               Alldispmeannegy = []
               
               Alldcrvar = []
               Allspeedvar = []
               Alldispvarpos = []
               Alldispvarneg = []
               
               Alldispvarposy = []
               Alldispvarnegy = []
               
               Alldispvarposx = []
               Alldispvarnegx = []
               
               for i in tqdm(range(starttime,endtime), total = endtime - starttime):
                         
                         
                         Curdcr = []
                         Curspeed = []
                         Curdisp = []
                         Curdispz = []
                         Curdispy = []
                         Curdispx = []
                         Curvec = []
                         for spotid, trackid, frameid in zip(self.AllValues[spotid_key], self.AllValues[trackid_key], self.AllValues[frameid_key]):
                     
                     
                         
                                 
                                 if i == int(frameid):
                                     try:
                                         dcr, speed, disp, zpos, ypos, xpos = Attr[int(spotid)]
                                         if dcr is not None:
                                           Curdcr.append(dcr)
                                           
                                         if speed is not None:
                                           Curspeed.append(speed)
                                         if disp is not None:
                                           Curdisp.append(disp)
                                         if zpos is not None:  
                                             Curdispz.append(zpos)
                                         if ypos is not None:
                                             Curdispy.append(ypos)
                                            
                                         if xpos is not None:
                                             Curdispx.append(xpos)
                                        
                                     except:
                                         
                                         pass
                               
                         dispZ = np.diff(Curdispz)
                         dispY = np.diff(Curdispy)
                         dispX = np.diff(Curdispx)
                         
                         meanCurdcr = np.mean(Curdcr)
                         varCurdcr = np.var(Curdcr)
                         if meanCurdcr is not None:
                           Alldcrmean.append(meanCurdcr)
                           Alldcrvar.append(varCurdcr)
                           Timedcr.append(i*self.tcalibration)
                           
                         meanCurspeed = np.mean(Curspeed)
                         varCurspeed = np.var(Curspeed)
                         if meanCurspeed is not None:
                             
                           Allspeedmean.append(meanCurspeed)
                           Allspeedvar.append(varCurspeed)
                           Timespeed.append(i*self.tcalibration)  
                           
                         meanCurdisp = np.mean(dispZ)
                         varCurdisp = np.var(dispZ)
                         
                         meanCurdispy = np.mean(dispY)
                         varCurdispy = np.var(dispY)
                         
                         meanCurdispx = np.mean(dispX)
                         varCurdispx = np.var(dispX)
                         
                         
                         if meanCurdisp is not None:
                           if meanCurdisp >=0: 
                              Alldispmeanpos.append(meanCurdisp)
                              Alldispvarpos.append(varCurdisp)
                              Timedisppos.append(i*self.tcalibration)
                           else:
                              Alldispmeanneg.append(meanCurdisp) 
                              Alldispvarneg.append(varCurdisp)
                              Timedispneg.append(i*self.tcalibration)
                           
                         
                             
                         if meanCurdispy is not None:
                           if meanCurdispy >=0: 
                              Alldispmeanposy.append(meanCurdispy)
                              Alldispvarposy.append(varCurdispy)
                              Timedispposy.append(i*self.tcalibration)
                           else:
                              Alldispmeannegy.append(meanCurdispy) 
                              Alldispvarnegy.append(varCurdispy)
                              Timedispnegy.append(i*self.tcalibration)   
                              
                         if meanCurdispx is not None:
                           if meanCurdispx >=0: 
                              Alldispmeanposx.append(meanCurdispx)
                              Alldispvarposx.append(varCurdispx)
                              Timedispposx.append(i*self.tcalibration)
                           else:
                              Alldispmeannegx.append(meanCurdispx) 
                              Alldispvarnegx.append(varCurdispx)
                              Timedispnegx.append(i*self.tcalibration)     
               
               
               plt.figure(figsize=(16, 8))
               plt.errorbar(Timespeed,Allspeedmean,Allspeedvar, linestyle='None',marker = '.', mfc = 'green', ecolor = 'green')
               plt.title('Speed')
               #plt.xlabel('Time (min)')
               #plt.ylabel('um/min')
               plt.savefig(self.savedir + '/' + "Speed", dpi = 300)
               plt.show()
               
               plt.figure(figsize=(16, 8))
               plt.errorbar(Timedisppos,Alldispmeanpos,Alldispvarpos, linestyle='None',marker = '.',  mfc = 'green', ecolor = 'green')
               plt.errorbar(Timedispneg,Alldispmeanneg,Alldispvarneg, linestyle='None',marker = '.',  mfc = 'red', ecolor = 'red')
               plt.title('Displacement in Z')
               #plt.xlabel('Time (min)')
               #plt.ylabel('um')
               plt.savefig(self.savedir + '/' + "Displacement_Z", dpi = 300)
               plt.show()
               
               plt.figure(figsize=(16, 8))
               plt.errorbar(Timedispposy,Alldispmeanposy,Alldispvarposy, linestyle='None',marker = '.',  mfc = 'green', ecolor = 'green')
               plt.errorbar(Timedispnegy,Alldispmeannegy,Alldispvarnegy, linestyle='None',marker = '.',  mfc = 'red', ecolor = 'red')
               plt.title('Displacement in Y')
               #plt.xlabel('Time (min)')
               #plt.ylabel('um')
               plt.savefig(self.savedir + '/' + "Displacement_Y", dpi = 300)
               plt.show()
               
               plt.figure(figsize=(16, 8))
               plt.errorbar(Timedispposx,Alldispmeanposx,Alldispvarposx, linestyle='None',marker = '.',  mfc = 'green', ecolor = 'green')
               plt.errorbar(Timedispnegx,Alldispmeannegx,Alldispvarnegx, linestyle='None',marker = '.',  mfc = 'red', ecolor = 'red')
               plt.title('Displacement in X')
               #plt.xlabel('Time (min)')
               #plt.ylabel('um')
               plt.savefig(self.savedir + '/' + "Displacement_X", dpi = 300)
               plt.show()
               
               
        def showNapari(self):
                 
                 self.viewer = napari.Viewer()
                 
                 self.viewer.add_labels(self.Segimage, name = self.Name)
                 for k in range(len(self.AllKeys)):
                            
                            if self.AllKeys[k] == 'POSITION_X':
                                self.keyX = k
                            if self.AllKeys[k] == 'POSITION_Y':
                                self.keyY = k   
                            if self.AllKeys[k] == 'POSITION_Z':
                                self.keyZ = k 
                            if self.AllKeys[k] == 'FRAME':
                                self.keyT = k    
                 
                 Attributeids = []
                 TrackAttributeids = []
                 for attributename in self.AllKeys:
                     Attributeids.append(attributename)
                 for attributename in self.AllTrackKeys:
                     TrackAttributeids.append(attributename)    
                 
                
                    
                 Attributeidbox = QComboBox()   
                 Attributeidbox.addItem(AttributeBoxname)   
                
                 TrackAttributeidbox = QComboBox()   
                 TrackAttributeidbox.addItem(TrackAttributeBoxname) 
                 computetrackbutton = QPushButton(' compute Track Relabelled Image')
                 computebutton = QPushButton(' compute Spot Relabelled Image')
                 for i in range(0, len(Attributeids)):
                     
                     
                     Attributeidbox.addItem(str(Attributeids[i]))
                     
                 for i in range(0, len(TrackAttributeids)):
                     
                     
                     TrackAttributeidbox.addItem(str(TrackAttributeids[i]))
                     
                     
                 
                 Attributeidbox.currentIndexChanged.connect(
                 lambda trackid = Attributeidbox: self.second_image_add(
                         
                         Attributeidbox.currentText(),
                         
                         self.Name,
                         False
                    
                )
            )           
                 
                 TrackAttributeidbox.currentIndexChanged.connect(
                 lambda trackid = TrackAttributeidbox: self.image_add(
                         
                         TrackAttributeidbox.currentText(),
                         
                         self.Name,
                         False
                    
                )
            )          
                 
                 computetrackbutton.clicked.connect(
                 lambda trackid = TrackAttributeidbox: self.image_add(
                         
                         TrackAttributeidbox.currentText(),
                         
                         self.Name,
                         True
                    
                )
            ) 
                 computebutton.clicked.connect(
                 lambda trackid = Attributeidbox: self.second_image_add(
                         
                         Attributeidbox.currentText(),
                         
                         self.Name,
                         True
                    
                )
            )   
                    
                
                 
                 self.viewer.window.add_dock_widget(Attributeidbox, name="Color Spot Attributes", area='right') 
                 self.viewer.window.add_dock_widget(TrackAttributeidbox, name="Color Track Attributes", area='right')
                 self.viewer.window.add_dock_widget(computetrackbutton, name="Compute Relabelled Image by Track features", area='left') 
                 self.viewer.window.add_dock_widget(computebutton, name="Compute Relabelled Image by Spot features", area='left') 
                 napari.run()
                 
                
        
                 
                        
        

        def image_add(self, attribute, imagename, compute = False ):

             if compute:
                          
                          self.idattr = {}
                          
                          for k in range(len(self.AllTrackKeys)):
                            
                            
                                if self.AllTrackKeys[k] == 'TRACK_ID':
                                       p = k
                                       
                                   
                          for k in range(len(self.AllTrackKeys)):
                            
                            self.AllTrackID = []
                            self.AllTrackAttr = []
                            if self.AllTrackKeys[k] ==  attribute:
                                
                                for attr, trackid in tqdm(zip(self.AllTrackValues[k], self.AllTrackValues[p]), total = len(self.AllTrackValues[k])):
                                    
                                            
                                            if math.isnan(trackid):
                                                continue
                                            else:
                                               self.idattr[trackid] = attr
                                               
                          for k in range(len(self.AllKeys)):
                            
                            locations = []
                            attrs = []
                            if self.AllKeys[k] ==  'TRACK_ID':
                                
                                for trackid, time, z, y, x in tqdm(zip(self.AllValues[k],self.AllValues[self.keyT],self.AllValues[self.keyZ],self.AllValues[self.keyY],self.AllValues[self.keyX] ), total = len(self.AllValues[k])):
                                        
                                       if len(self.Segimage.shape) == 4:
                                           centroid = (time, z, y, x)
                                       else:
                                           centroid = (time, y, x) 
                                       try:
                                         attr = self.idattr[trackid]
                                         locations.append([attr, centroid]) 
                                       except:
                                           pass
                                           
                                self.Segimage = self.Relabel(self.Segimage.copy(), locations)
                                for layer in list(self.viewer.layers):
                                   if  layer.name in self.Name or self.Name in layer.name:
                                       self.viewer.layers.remove(layer) 
                                self.viewer.add_labels(self.Segimage, name = self.Name + attribute)
                 

                             
        def second_image_add(self, attribute, imagename, compute = False):
                        
                        if compute:
                          
                          self.boxes = {}
                          self.labels = {}
                          
                          for k in range(len(self.AllKeys)):
                            
                            locations = []
                            attrs = []
                            if self.AllKeys[k] ==  attribute:
                                
                                for attr, time, z, y, x in tqdm(zip(self.AllValues[k],self.AllValues[self.keyT],self.AllValues[self.keyZ],self.AllValues[self.keyY],self.AllValues[self.keyX] ), total = len(self.AllValues[k])):
                                       if len(self.Segimage.shape) == 4:
                                           centroid = (time, z, y, x)
                                       else:
                                           centroid = (time, y, x) 
                                            
                                       locations.append([attr, centroid]) 
                                       
                                NewSegimage = self.Relabel(self.Segimage.copy(), locations)
                                self.viewer.add_labels(NewSegimage, name = self.Name + attribute)  
                             
                    
        def Relabel(self, image, locations):
        
               print("Relabelling image with chosen trackmate attribute")
               NewSegimage = image.copy()
               for p in tqdm(range(0, NewSegimage.shape[0])):
                   
                   sliceimage = NewSegimage[p,:]
                   originallabels = []
                   newlabels = []
                   for  relabelval, centroid in locations:
                        if len(NewSegimage.shape) == 4: 
                            time, z, y, x = centroid
                        else:
                            time, y, x = centroid 
                        
                        if p == int(time): 
                               
                               if len(NewSegimage.shape) == 4:  
                                  originallabel = sliceimage[z,y,x]
                               else:
                                   originallabel = sliceimage[y,x]
                                    
                               if originallabel ==0: relabelval = 0
                               if math.isnan(relabelval):
                                      relabelval = -1
                               originallabels.append(int(originallabel))
                               newlabels.append(int(relabelval))
                      
                   relabeled = map_array(sliceimage, np.asarray(originallabels), np.asarray(newlabels))
                   NewSegimage[p,:] = relabeled
                               
               return NewSegimage                     
               
              
                    
            

def import_TM_XML(xml_path, image, Segimage = None, Mask=None):
    
    image = imread(image)
    if Segimage is not None:
        Segimage = imread(Segimage)
    if Mask is not None:
        Mask = imread(Mask)

    root = et.fromstring(codecs.open(xml_path, 'r', 'utf8').read())

    filtered_track_ids = [
        int(track.get('TRACK_ID'))
        for track in root.find('Model').find('FilteredTracks').findall('TrackID')
    ]

    # Extract the tracks from xml
    tracks = root.find('Model').find('AllTracks')
    settings = root.find('Settings').find('ImageData')

    # Extract the cell objects from xml
    Spotobjects = root.find('Model').find('AllSpots')

    # Make a dictionary of the unique cell objects with their properties
    Uniqueobjects = {}
    Uniqueproperties = {}

    xcalibration = float(settings.get('pixelwidth'))
    ycalibration = float(settings.get('pixelheight'))
    zcalibration = float(settings.get('voxeldepth'))
    tcalibration = int(float(settings.get('timeinterval')))

    if Mask is not None:
        if len(Mask.shape) < len(image.shape):
            # T Z Y X
            UpdateMask = np.zeros(
                [
                    image.shape[0],
                    image.shape[1],
                    image.shape[2],
                    image.shape[3],
                ]
            )
            for i in range(0, UpdateMask.shape[0]):
                for j in range(0, UpdateMask.shape[1]):

                    UpdateMask[i, j, :, :] = Mask[i, :, :]
                    
        else:
            UpdateMask = Mask
            
            
        Mask = UpdateMask.astype('uint16')

        TimedMask, Boundary = boundary_points(Mask, xcalibration, ycalibration, zcalibration)
    else:
        TimedMask = None
        Boundary = None

    for frame in Spotobjects.findall('SpotsInFrame'):

        for Spotobject in frame.findall('Spot'):
            # Create object with unique cell ID
            cell_id = int(Spotobject.get("ID"))
            # Get the TZYX location of the cells in that frame
            Uniqueobjects[cell_id] = [
                Spotobject.get('FRAME'),
                Spotobject.get('POSITION_Z'),
                Spotobject.get('POSITION_Y'),
                Spotobject.get('POSITION_X'),
            ]
            
            # Get other properties associated with the Spotobject
            try:
                TOTAL_INTENSITY_CH1 = Spotobject.get('TOTAL_INTENSITY_CH1')
            except:
                 
                 TOTAL_INTENSITY_CH1 = 1
            try:
                MEAN_INTENSITY_CH1 = Spotobject.get('MEAN_INTENSITY_CH1')
            except:
                 MEAN_INTENSITY_CH1 = 1
            try:      
                Radius = Spotobject.get('RADIUS')
            except:
                Radius = 1
                
            try:      
                QUALITY = Spotobject.get('QUALITY')
            except:
                QUALITY = 1
                    
            Uniqueproperties[cell_id] = [
                
                  TOTAL_INTENSITY_CH1,
                  MEAN_INTENSITY_CH1,
                Spotobject.get('FRAME'),
                Radius,
                QUALITY
            ]
    all_track_properties = []

    x_ls = tracks.findall('Track')
    print('total tracks',len(x_ls))
    for track in x_ls:

         track_id = int(track.get("TRACK_ID"))

         if track_id in filtered_track_ids:
            tracklets, DividingTrajectory, split_points_times = track_function(track, track_id, filtered_track_ids,Uniqueproperties)
    
 
            location_prop_dist = tracklet_properties(
                tracklets,
                Uniqueobjects,
                Uniqueproperties,
                Mask,
                Segimage,
                TimedMask,
                [xcalibration, ycalibration, zcalibration, tcalibration],
                DividingTrajectory
            )
            all_track_properties.append([track_id, location_prop_dist, DividingTrajectory])
   
    return all_track_properties, split_points_times, Boundary, [
        xcalibration,
        ycalibration,
        zcalibration,
        tcalibration,
    ]

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def track_function(track, track_id, filtered_track_ids, Uniqueproperties):
  
        

        spot_object_source_target = []
        
        split_points_times = []
        
        for edge in track.findall('Edge'):

                source_id = edge.get('SPOT_SOURCE_ID')
                target_id = edge.get('SPOT_TARGET_ID')
                try:
                   TOTAL_INTENSITY_CH1, MEAN_INTENSITY_CH1,Position_T,Radius,QUALITY = Uniqueproperties[int(source_id)]
                except:
                   TOTAL_INTENSITY_CH1, MEAN_INTENSITY_CH1,Position_T,Radius,QUALITY = Uniqueproperties[int(target_id)]
                
                edge_time = Position_T
                
                directional_rate_change = edge.get('DIRECTIONAL_CHANGE_RATE')
                speed = edge.get('SPEED')

                spot_object_source_target.append(
                    [source_id, target_id, edge_time, directional_rate_change, speed]
                )

            # Sort the tracks by edge time
        spot_object_source_target = sorted(
                spot_object_source_target, key=sortTracks, reverse=False
            )
            # Get all the IDs, uniquesource, targets attached, leaf, root, splitpoint IDs
        split_points, split_times, root_leaf = Multiplicity(spot_object_source_target)
            
            # Determine if a track has divisions or none
        if len(split_points) > 0:
                split_points = split_points[::-1]
                DividingTrajectory = True
        else:
                DividingTrajectory = False
        
        if DividingTrajectory == True:
                DividingTrackIds.append(track_id)
                AllTrackIds.append(track_id)
                tracklets = analyze_dividing_tracklets(
                    root_leaf, split_points, spot_object_source_target
                )
                for i in range(len(split_points)):
                      split_points_times.append([split_points[i], split_times[i]])
                      
        if DividingTrajectory == False:
                NonDividingTrackIds.append(track_id)
                AllTrackIds.append(track_id)
                tracklets = analyze_non_dividing_tracklets(
                    root_leaf, spot_object_source_target
                )

        return tracklets, DividingTrajectory, split_points_times 

def common_stats_function(xml_path, image = None, Mask = None):
    
    if image is not None:
      image = daskread(image)[0,:]
    if Mask is not None:
        Mask = imread(Mask)
    root = et.fromstring(codecs.open(xml_path, 'r', 'utf8').read())

    filtered_track_ids = [
        int(track.get('TRACK_ID'))
        for track in root.find('Model').find('FilteredTracks').findall('TrackID')
    ]
    

    # Extract the tracks from xml
    tracks = root.find('Model').find('AllTracks')
    settings = root.find('Settings').find('ImageData')

    # Extract the cell objects from xml
    Spotobjects = root.find('Model').find('AllSpots')

    # Make a dictionary of the unique cell objects with their properties
    Uniqueobjects = {}
    Uniqueproperties = {}

    xcalibration = float(settings.get('pixelwidth'))
    ycalibration = float(settings.get('pixelheight'))
    zcalibration = float(settings.get('voxeldepth'))
    tcalibration = int(float(settings.get('timeinterval')))

    plotMask = Mask
    if  Mask is not None:
        if len(Mask.shape) < len(image.shape):
            # T Z Y X
            plotMask = Mask
            UpdateMask = np.zeros(
                [
                    image.shape[0],
                    image.shape[1],
                    image.shape[2],
                    image.shape[3],
                ]
            )
            for i in range(0, UpdateMask.shape[0]):
                for j in range(0, UpdateMask.shape[1]):

                    UpdateMask[i, j, :, :] = Mask[i, :, :]
                    
        else:
            UpdateMask = Mask
            
        Mask = UpdateMask.astype('uint16')

        TimedMask, Boundary = boundary_points(Mask, xcalibration, ycalibration, zcalibration)
    else:
        TimedMask = None
        Boundary = None    
            
   

    for frame in Spotobjects.findall('SpotsInFrame'):

        for Spotobject in frame.findall('Spot'):
            # Create object with unique cell ID
            cell_id = int(Spotobject.get("ID"))
            # Get the TZYX location of the cells in that frame
            
            Uniqueobjects[cell_id] = [
                Spotobject.get('FRAME'),
                Spotobject.get('POSITION_Z'),
                Spotobject.get('POSITION_Y'),
                Spotobject.get('POSITION_X'),
            ]
            
            # Get other properties associated with the Spotobject
            try:
                TOTAL_INTENSITY_CH1 = Spotobject.get('TOTAL_INTENSITY_CH1')
            except:
                 
                 TOTAL_INTENSITY_CH1 = 1
            try:
                MEAN_INTENSITY_CH1 = Spotobject.get('MEAN_INTENSITY_CH1')
            except:
                 MEAN_INTENSITY_CH1 = 1
            try:      
                Radius = Spotobject.get('RADIUS')
            except:
                Radius = 1
                
            try:      
                QUALITY = Spotobject.get('QUALITY')
            except:
                QUALITY = 1
                    
            Uniqueproperties[cell_id] = [
                
                  TOTAL_INTENSITY_CH1,
                  MEAN_INTENSITY_CH1,
                Spotobject.get('FRAME'),
                Radius,
                QUALITY
            ]

    all_track_properties = []
    split_points_times = []
    
    for track in tracks.findall('Track'):

        track_id = int(track.get("TRACK_ID"))

        spot_object_source_target = []
        if track_id in filtered_track_ids:
            for edge in track.findall('Edge'):

                source_id = edge.get('SPOT_SOURCE_ID')
                target_id = edge.get('SPOT_TARGET_ID')
                try:
                   TOTAL_INTENSITY_CH1, MEAN_INTENSITY_CH1,Position_T,Radius,QUALITY = Uniqueproperties[int(source_id)]
                except:
                   TOTAL_INTENSITY_CH1, MEAN_INTENSITY_CH1,Position_T,Radius,QUALITY = Uniqueproperties[int(target_id)] 
                edge_time = Position_T
                directional_rate_change = edge.get('DIRECTIONAL_CHANGE_RATE')
                speed = edge.get('SPEED')

                spot_object_source_target.append(
                    [source_id, target_id, edge_time, directional_rate_change, speed]
                )

            # Sort the tracks by edge time
            spot_object_source_target = sorted(
                spot_object_source_target, key=sortTracks, reverse=False
            )
            # Get all the IDs, uniquesource, targets attached, leaf, root, splitpoint IDs
            split_points, split_times, root_leaf = Multiplicity(spot_object_source_target)

            # Determine if a track has divisions or none
            if len(split_points) > 0:
                split_points = split_points[::-1]
                DividingTrajectory = True
            else:
                DividingTrajectory = False
          
            if DividingTrajectory == True:
                DividingTrackIds.append(track_id)
                AllTrackIds.append(track_id)
                tracklets = analyze_dividing_tracklets(
                    root_leaf, split_points, spot_object_source_target
                )
                for i in range(len(split_points)):
                      split_points_times.append([split_points[i], split_times[i]])
            if DividingTrajectory == False:
                NonDividingTrackIds.append(track_id)
                AllTrackIds.append(track_id)
                tracklets = analyze_non_dividing_tracklets(
                    root_leaf, spot_object_source_target
                )

            # for each tracklet get real_time,z,y,x,total_intensity, mean_intensity, cellradius, distance, prob_inside
            location_prop_dist = tracklet_properties(
                tracklets,
                Uniqueobjects,
                Uniqueproperties,
                Mask,
                None,
                TimedMask,
                [xcalibration, ycalibration, zcalibration, tcalibration],
                DividingTrajectory
            )
            all_track_properties.append([track_id, location_prop_dist, DividingTrajectory])
            
    return all_track_properties, split_points_times, TimedMask, Boundary, xcalibration, ycalibration, zcalibration, tcalibration , image, Mask, plotMask       
            
    
def import_TM_XML_Randomization(xml_path,image = None, Mask = None, nbins = 5):
    
    all_track_properties, split_points_times, TimedMask, Boundary, xcalibration, ycalibration, zcalibration, tcalibration, image, Mask, plotMask = common_stats_function(xml_path, image, Mask)        
    TrackLayerTracklets = {}
    AllT = []
    
        
    
    for i in tqdm(range(0, len(all_track_properties))):
                    trackid, alltracklets, DividingTrajectory = all_track_properties[i]
                    TrackIDLocations = []
                    TrackLayerTracklets[trackid] = [trackid]
                    for (trackletid, tracklets) in alltracklets.items():
                            Locationtracklets = tracklets[1]
                            if len(Locationtracklets) > 0:
                                Locationtracklets = sorted(
                                    Locationtracklets, key=sortFirst, reverse=False
                                )
        
                                for tracklet in Locationtracklets:
                                    (
                                        t,
                                        z,
                                        y,
                                        x,
                                        total_intensity,
                                        mean_intensity,
                                        cellradius,
                                        distance,
                                        prob_inside,
                                        speed,
                                        directionality,
                                        DividingTrajectory,
                                    ) = tracklet
                                    
                                    TrackIDLocations.append([float(t), float(z), float(y), float(x)])
        
                    TrackLayerTracklets[trackid].append(TrackIDLocations)
                    
    for (trackid, tracklets) in TrackLayerTracklets.items():

            
            locations = tracklets[1]
            
            df = pd.DataFrame(locations)
            #Create deltat, deltaz, deltay, deltax
            derivativedf = df.diff()
            t = df[0][1:]
            deltat = derivativedf[0][1:] 
            deltaz = derivativedf[1][1:] 
            deltay = derivativedf[2][1:] 
            deltax = derivativedf[3][1:] 
            
            print('Gaussfits for trackid: ', trackid)
            gmodel = Model(gaussian)
            meanz, stdz = norm.fit(deltaz) 
            countsz, binsz = np.histogram(deltaz,bins = nbins )
            binsz = binsz[:-1]
            try:
              GaussZ = gmodel.fit(countsz, x=binsz, amp=np.max(countsz), mu=meanz, std=stdz)
            except:
                GaussZ = None
            meany, stdy = norm.fit(deltay) 
            countsy, binsy = np.histogram(deltay, bins = nbins)
            binsy = binsy[:-1]
            try:
              GaussY = gmodel.fit(countsy, x=binsy, amp=np.max(countsy), mu=meany, std=stdy)
            except:
              GaussY = None  
                
            meanx, stdx = norm.fit(deltax) 
            countsx, binsx = np.histogram(deltax, bins = nbins)
            binsx = binsx[:-1]
            
            try:
               GaussX = gmodel.fit(countsx, x=binsx, amp=np.max(countsx), mu=meanx, std=stdx)
            except:
                GaussX = None
            tripleplot(binsz, binsy, binsx, countsz, countsy, countsx, GaussZ, GaussY, GaussX)
            
            if GaussZ is not None:
               print('Fit in Z', GaussZ.fit_report())
            if GaussY is not None:   
               print('Fit in Y', GaussY.fit_report())
            if GaussX is not None:     
              print('Fit in X', GaussX.fit_report())
                     
                   
 
 
def tripleplot(binsz, binsy, binsx, countsz, countsy, countsx, GaussZ, GaussY, GaussX):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].hist(binsz,binsz, weights = countsz)
    if GaussZ is not None:
        ax[0].plot(binsz,GaussZ.best_fit,'ro:',label='fit')
        ax[0].set_title('Gaussian Fit')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('DisplacementZ')
    
    ax[1].hist(binsy,binsy, weights = countsy)
    
    if GaussY is not None:
        ax[1].plot(binsy,GaussY.best_fit,'ro:',label='fit')
        ax[1].set_title('Gaussian Fit')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('DisplacementY')
    
    ax[2].hist(binsx,binsx, weights = countsx)
    
    if GaussX is not None:
        ax[2].plot(binsx,GaussX.best_fit,'ro:',label='fit')
        ax[2].set_title('Gaussian Fit')
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('DisplacementX')
    
    
    plt.tight_layout()
    plt.show()

  
def plot_3D_polylines_xy(polyline_df_xy, t, ax, line_color, line_alpha):
    tMin, tMax = polyline_df_xy.t.min(), polyline_df_xy.t.max()
    xMin, xMax = polyline_df_xy.x.min(), polyline_df_xy.x.max()
    yMin, yMax = polyline_df_xy.y.min(), polyline_df_xy.y.max()
    zMin, zMax = polyline_df_xy.z.min(), polyline_df_xy.z.max()
    shift_x, shift_y, shift_z = np.mean([xMax, xMin]), np.mean([yMax, yMin]), np.mean([zMax, zMin])
    
    polyline_df_xy_t = polyline_df_xy.loc[polyline_df_xy.t == t]
    # plot the horizontal lines pf mesh
    for i in polyline_df_xy_t.z.unique():
        temp = polyline_df_xy_t.loc[polyline_df_xy_t.z == i]
        x, y, z = temp.x.values, temp.y.values, temp.z.values
        ax.plot(x-shift_x, y-shift_y, z-shift_z, '-', color=line_color, alpha=line_alpha, lw=0.2)

def plot_3D_polylines_yz(polyline_df_yz, t, ax, line_color, line_alpha):
    tMin, tMax = polyline_df_yz.t.min(), polyline_df_yz.t.max()
    xMin, xMax = polyline_df_yz.x.min(), polyline_df_yz.x.max()
    yMin, yMax = polyline_df_yz.y.min(), polyline_df_yz.y.max()
    zMin, zMax = polyline_df_yz.z.min(), polyline_df_yz.z.max()
    shift_x, shift_y, shift_z = np.mean([xMax, xMin]), np.mean([yMax, yMin]), np.mean([zMax, zMin])
    
    polyline_df_yz_t = polyline_df_yz.loc[polyline_df_yz.t == t]
    # plot the horizontal lines pf mesh
    for i in polyline_df_yz_t.z.unique():
        temp = polyline_df_yz_t.loc[polyline_df_yz_t.z == i]
        x, y, z = temp.x.values, temp.y.values, temp.z.values
        ax.plot(z-shift_z, y-shift_y, x-shift_x, '-', color=line_color, alpha=line_alpha, lw=0.2)
  
def import_TM_XML_Localization(xml_path,image = None, Mask = None, window_size = 5, angle_1 = 45, angle_2 = 60):
    print('Reading XML')
    all_track_properties, split_points_times, TimedMask, Boundary, xcalibration, ycalibration, zcalibration, tcalibration, image, Mask, plotMask = common_stats_function(xml_path, image, Mask)
    print('Done, Processing')
    IDLocations = []
    TrackLayerTracklets = {}
    Gradients = []
    AllT = []
    figure = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection='3d')
    
    figuressec = plt.figure(figsize=(16, 10))
    axdir = plt.axes(projection='3d')
    plt.autoscale(enable = True)
    
    figure2D = plt.figure(figsize=(16, 10))
    multiplot_widget = FigureCanvas(figure2D)
    ax2D = multiplot_widget.figure.subplots(1, 4)
    
    
    figure2Dnext = plt.figure(figsize=(16, 10))
    multiplot_widget_next = FigureCanvas(figure2Dnext)
    ax2D_next = multiplot_widget_next.figure.subplots(1, 2)
    
    # Data for a three-dimensional line
    print('All Tracks plot') 
    if TimedMask is not None:
        
        tree, indices, masklabel, masklabelvolume = TimedMask[str(int(float(0)))]
       
        MaskYs = []
        MaskXs = []
        count = 0
        for z,y,x in zip(*indices):
            if count ==0:
                y0 = y
                x0 = x
            MaskYs.append(y - y0)
            MaskXs.append(x - x0)
            count = count + 1
        tree, indices, masklabel, masklabelvolume = TimedMask[str(int(float(Mask.shape[0] - 1)))]
       
        MaskYf = []
        MaskXf = []
        count = 0
        for z,y,x in zip(*indices):
            if count ==0:
                y0 = y
                x0 = x
            MaskYf.append(y - y0)
            MaskXf.append(x - x0)     
            count = count + 1
            
    SuperTracksT = []
    SuperAnglesTime = []   
    SuperAngles = []      
    for i in tqdm(range(0, len(all_track_properties))):
                    trackid, alltracklets, DividingTrajectory = all_track_properties[i]

                    
                    AllDistance = []
                    AllTracksZ = []
                    AllTracksY = []
                    AllTracksX = []
                    AllTracksT = []
                    AllAngles = []
                    AllAnglesTime = []
                    AllAnglesTrackID = []
                    AllT.append(trackid)
                    for (trackletid, tracklets) in alltracklets.items():
                            Locationtracklets = tracklets[1]
                            if len(Locationtracklets) > 0:
                                Locationtracklets = sorted(
                                    Locationtracklets, key=sortFirst, reverse=False
                                )
        
                                for tracklet in Locationtracklets:
                                    (
                                        t,
                                        z,
                                        y,
                                        x,
                                        total_intensity,
                                        mean_intensity,
                                        cellradius,
                                        distance,
                                        prob_inside,
                                        speed,
                                        directionality,
                                        DividingTrajectory,
                                    ) = tracklet
                                    
                                    
                                    AllDistance.append((float(distance)))
                                    AllTracksZ.append(float(z))
                                    AllTracksY.append(float(y))
                                    AllTracksX.append(float(x))
                                    AllTracksT.append(float(t))
                                    SuperTracksT.append(float(t))
                    xMax = np.max(AllTracksX)
                    yMax = np.max(AllTracksY)
                    zMax = np.max(AllTracksZ)   
                    
                    xMin = np.min(AllTracksX)
                    yMin = np.min(AllTracksY)
                    zMin = np.min(AllTracksZ)
                    
                                
                    shift_x, shift_y, shift_z = np.mean([xMax, xMin]), np.mean([yMax, yMin]), np.mean([zMax, zMin])
                    
                    scounts = []
                    scounts_time = []
                    
                    
                    for i in range(0,len(AllTracksT)):
                         
                         
                         if i > 0:
                            j = i + window_size
                         if i == 0:
                             j = 0
                         
                         if len(AllTracksZ) > j + window_size:
                                 z = AllTracksZ[j]
                                 x = AllTracksX[j]
                                 y = AllTracksY[j]
                                 vector_1 = [z,y,x]
                                 nexttime = AllTracksT[j + window_size]
                                 nextz = AllTracksZ[j + window_size]
                                 nextx = AllTracksX[j + window_size]  
                                 nexty = AllTracksY[j + window_size] 
                                 vector_2 = [nextz,nexty,nextx]
                                 unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                                 unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                                 dot_product = np.dot(unit_vector_1, unit_vector_2)
                                 angle = np.arccos(dot_product) * 180 / np.pi
                                 AllAngles.append(float(angle))
                                 AllAnglesTime.append(float(nexttime))
                                 
                                 SuperAnglesTime.append(float(nexttime))
                                 SuperAngles.append(float(angle))
                                 
                                 AllAnglesTrackID.append(float(trackid))
                                 
                    # AllTracksZ = np.diff(AllTracksZ)
                    # AllTracksY = np.diff(AllTracksY)
                    # AllTracksX = np.diff(AllTracksX)
                    
                    # AllTracksZ = MovingAverage(
                    #             AllTracksZ, window_size=window_size
                    #         )
                    
                    # AllTracksY = MovingAverage(
                    #             AllTracksY, window_size=window_size
                    #         )
                    
                    # AllTracksX = MovingAverage(
                    #             AllTracksX, window_size=window_size
                    #         )
                    
                    
                    gradient = np.sum(np.diff(AllDistance))
                    Gradients.append(gradient)
      

                    line_color = '#1A6A82'
                    line_alpha = 0.8
                    lw = 0.4
                    ax.view_init(angle_1, angle_2)
                    ax.plot3D(AllTracksX - shift_x, AllTracksY - shift_y, AllTracksZ - shift_z, '-',  alpha=line_alpha, lw=lw )
                    
                    ax.set_xlabel('dx')
                    ax.set_ylabel('dy')
                    ax.set_zlabel('dz')
                    
                    axdir.view_init(angle_1, angle_2)
                    axdir.plot3D(AllAnglesTrackID, AllAnglesTime, AllAngles, '-',  alpha=line_alpha, lw=lw ) 
                    
                    axdir.set_xlabel('trackid')
                    axdir.set_ylabel('time')
                    axdir.set_zlabel('angles')
                    
                    ax2D[0].plot(AllTracksX - shift_x, AllTracksY - shift_y, '-',  alpha=line_alpha, lw=lw )
                    ax2D[0].set_xlabel('dx')
                    ax2D[0].set_ylabel('dy')
                    
                    ax2D[1].plot(AllTracksZ - shift_z, AllTracksY - shift_y, '-',  alpha=line_alpha, lw=lw )
                    ax2D[1].set_xlabel('dz')
                    ax2D[1].set_ylabel('dy')
                    
                    ax2D[2].plot(AllTracksZ - shift_z, AllTracksX - shift_x, '-',  alpha=line_alpha, lw=lw )
                    ax2D[2].set_xlabel('dz')
                    ax2D[2].set_ylabel('dx')
                    
                    
    Alltimes = sorted(set(SuperTracksT.copy()))             
    for i in range(0,len(Alltimes)):
         
         timenow = Alltimes[i]
         
         scount = 0
         for j in range(0, len(split_points_times)):
             
             split_point, split_time = split_points_times[j]
             split_time = int(split_time)
             if split_time == int(timenow):
                 scount = scount + 1
             
         scounts_time.append(float(timenow))
         scounts.append(float(scount))
    SetAllAnglesTime = sorted(set(SuperAnglesTime.copy()))   
    AverageAngle = []
    AverageTime = []
    for i in range(0, len(SetAllAnglesTime)):
        
        average_angle = 0
        count = 0
        time = SetAllAnglesTime[i]
        for j in range(0, len(SuperAnglesTime)):
            
            nexttime = SuperAnglesTime[j]
            
            if nexttime == time:
                average_angle = average_angle + SuperAngles[j]
                count = count + 1
        AverageAngle.append(average_angle/count)
        AverageTime.append(time)

            
    ax2D_next[0].plot(scounts_time, scounts, '-',  alpha=line_alpha, lw=lw )
    ax2D_next[0].set_xlabel('time (frames)')
    ax2D_next[0].set_ylabel('division')
                  
    ax2D_next[1].plot(AverageTime, AverageAngle, '-',  alpha=line_alpha, lw=lw )
    ax2D_next[1].set_xlabel('time (frames)')
    ax2D_next[1].set_ylabel('angles')          
                  
    sns.histplot(Gradients, kde = True, ax = ax2D[3])
    
  
                   
    return Gradients            
            
            
            
def Multiplicity(spot_object_source_target):

    split_points = []
    split_times = []
    root_leaf = []
    sources = []
    targets = []
    scount = 0

    for (
        source_id,
        target_id,
        sourcetime,
        directional_rate_change,
        speed,
    ) in spot_object_source_target:

        sources.append(source_id)
        targets.append(target_id)

    root_leaf.append(sources[0])
    for (
        source_id,
        target_id,
        sourcetime,
        directional_rate_change,
        speed,
    ) in spot_object_source_target:

        if target_id not in sources:

            root_leaf.append(target_id)

        for (
            sec_source_id,
            sec_target_id,
            sec_sourcetime,
            sec_directional_rate_change,
            sec_speed,
        ) in spot_object_source_target:
            if source_id == sec_source_id:
                scount = scount + 1
        if scount > 1:
            split_points.append(source_id)
            split_times.append(sourcetime)
        scount = 0

    return split_points, split_times, root_leaf






def speed_labels(bins, units):   
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == bins[0]:
            labels.append('calm'.format(right))
        elif np.isinf(right):
            labels.append('>{} {}'.format(left, units))
        else:
            labels.append('{} - {} {}'.format(left, right, units))

    return list(labels)        
     
def _convert_dir(directions, N=None):
    if N is None:
        N = directions.shape[0]
    barDir = directions * pi/180. - pi/N
    barWidth = 2 * pi / N
    return barDir, barWidth        


def wind_rose(rosedata, wind_dirs, palette=None):
    if palette is None:
        palette = sns.color_palette('inferno', n_colors=rosedata.shape[1])

    bar_dir, bar_width = _convert_dir(wind_dirs)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
        if n == 0:
            # first column only
            ax.bar(bar_dir, rosedata[c1].values, 
                   width=bar_width,
                   color=palette[0],
                   edgecolor='none',
                   label=c1,
                   linewidth=0)

        # all other columns
        ax.bar(bar_dir, rosedata[c2].values, 
               width=bar_width, 
               bottom=rosedata.cumsum(axis=1)[c1].values,
               color=palette[n+1],
               edgecolor='none',
               label=c2,
               linewidth=0)

    leg = ax.legend(loc=(0.75, 0.95), ncol=2)
    xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
    return fig

class AllTrackViewerGauss(object):
    def __init__(
        self,
        originalviewer,
        Raw,
        Seg,
        Mask,
        savedir,
        calibration,
        all_track_properties,
        ID,
        canvas,
        ax,
        
        figure,
        DividingTrajectory = None,
        saveplot=False,
        nbins = 10,
        window_size=3,
        mode='fate',
        
    ):

        self.trackviewer = originalviewer
        self.savedir = savedir
        Path(self.savedir).mkdir(exist_ok=True)
        self.Raw = Raw
        self.Seg = Seg
        self.Mask = Mask
        self.nbins = nbins
        self.calibration = calibration
        self.mode = mode
        if ID == 'all':
            self.ID = ID
        elif ID is not None:
            self.ID = int(ID)
        else:
            self.ID = ID
        self.saveplot = saveplot
        self.canvas = canvas
        self.ax = ax
        self.window_size = window_size
        self.figure = figure
        self.all_track_properties = all_track_properties
        self.DividingTrajectory = DividingTrajectory
        self.tracklines = 'Tracks'
        self.boxes = {}
        self.labels = {}
        for layer in list(self.trackviewer.layers):

            if self.tracklines in layer.name or layer.name in self.tracklines:
                self.trackviewer.layers.remove(layer)
        self.draw()
        if self.mode == 'fate':
            self.plot()
        
    def plot(self):

        for i in range(self.ax.shape[0]):
            for j in range(self.ax.shape[1]):
                  self.ax[i,j].cla()

        self.ax[0,0].set_title("distance_to_boundary")
        self.ax[0,0].set_xlabel("minutes")
        self.ax[0,0].set_ylabel("um")

        self.ax[0,1].set_title("cell_tissue_localization")
        self.ax[0,1].set_xlabel("start_distance")
        self.ax[0,1].set_ylabel("end_distance")
        
        
        self.ax[1,0].set_title("Displacementz")
        self.ax[1,0].set_xlabel("deltaz")
        self.ax[1,0].set_ylabel("countsz")

        self.ax[1,1].set_title("Displacementy")
        self.ax[1,1].set_xlabel("deltay")
        self.ax[1,1].set_ylabel("countsy")

        self.ax[1,2].set_title("Displacementx")
        self.ax[1,2].set_xlabel("deltax")
        self.ax[1,2].set_ylabel("countsx")

        # Execute the function
        TrackLayerTracklets = {}
        for i in tqdm(range(0, len(self.all_track_properties))):
                
                    IDLocations = []
                    trackid, alltracklets, DividingTrajectory = self.all_track_properties[i]
                    if self.ID == trackid or self.ID == 'all':
                        tracksavedir = self.savedir + '/' + 'TrackID' +  str(self.ID)
                        Path(tracksavedir).mkdir(exist_ok=True)
                        AllStartParent[trackid] = [trackid]
                        AllEndParent[trackid] = [trackid]
        
                        TrackLayerTracklets[trackid] = [trackid]
                        for (trackletid, tracklets) in alltracklets.items():
        
                            AllStartChildren[int(str(trackid) + str(trackletid))] = [
                                int(str(trackid) + str("_") +str(trackletid))
                            ]
                            AllEndChildren[int(str(trackid) + str(trackletid))] = [
                                int(str(trackid) +str("_") + str(trackletid))
                            ]
        
                            self.AllT = []
                            self.AllArea = []
                            self.AllIntensity = []
                            self.AllProbability = []
                            self.AllSpeed = []
                            self.AllDirection = []
                            self.AllSize = []
                            self.AllDistance = []
                            TrackLayerTrackletsList = []
        
                            Locationtracklets = tracklets[1]
                            if len(Locationtracklets) > 0:
                                Locationtracklets = sorted(
                                    Locationtracklets, key=sortFirst, reverse=False
                                )
        
                                for tracklet in Locationtracklets:
                                    (
                                        t,
                                        z,
                                        y,
                                        x,
                                        total_intensity,
                                        mean_intensity,
                                        cellradius,
                                        distance,
                                        prob_inside,
                                        speed,
                                        directionality,
                                        DividingTrajectory,
                                    ) = tracklet
                                    TrackLayerTrackletsList.append([trackletid, t, z, y, x])
                                    IDLocations.append([float(t), float(z), float(y), float(x)])
                                    
                                    self.AllT.append(int(float(t )))
                                    self.AllSpeed.append("{:.1f}".format(float(speed)))
                                    self.AllDirection.append("{:.1f}".format(float(directionality)))
                                    self.AllProbability.append(
                                        "{:.2f}".format(float(prob_inside))
                                    )
                                    self.AllDistance.append("{:.1f}".format(float(distance)))
                                    self.AllSize.append("{:.1f}".format(float(cellradius)))
                                    if str(self.ID) + str(trackletid) not in AllID:
                                        AllID.append(str(self.ID) + str(trackletid))
                                if trackletid == 0:
                                    if len(self.AllDistance) > 1:
                                            AllStartParent[trackid].append(float(self.AllDistance[0]))
                                            AllEndParent[trackid].append(float(self.AllDistance[-1]))
        
                                else:
                                    if len(self.AllDistance) > 1:
                                            AllStartChildren[
                                                int(str(trackid) + str("_") +  str(trackletid))
                                            ].append(float(self.AllDistance[0]))
                                            AllEndChildren[int(str(trackid) +str("_") + str(trackletid))].append(
                                                float(self.AllDistance[-1])
                                            )
        
                            self.AllSpeed = MovingAverage(
                                self.AllSpeed, window_size=self.window_size
                            )
                            self.AllDirection = MovingAverage(
                                self.AllDirection, window_size=self.window_size
                            )
                            self.AllDistance = MovingAverage(
                                self.AllDistance, window_size=self.window_size
                            )
                            self.AllSize = MovingAverage(
                                self.AllSize, window_size=self.window_size
                            )
                            self.AllProbability = MovingAverage(
                                self.AllProbability, window_size=self.window_size
                            )
                            self.AllT = MovingAverage(self.AllT, window_size=self.window_size)
                            if self.saveplot == True:
                                SaveIds.append(self.ID)
                                
                                df = pd.DataFrame(
                                    list(
                                        zip(
                                            self.AllT,
                                            self.AllSize,
                                            self.AllDistance,
                                            self.AllProbability,
                                            self.AllSpeed,
                                            self.AllDirection
                                        )
                                    ),
                                    columns=[
                                        'Time',
                                        'Cell Size',
                                        'Distance to Border',
                                        'Inner Cell Probability',
                                        'Cell Speed',
                                        'Directionality'
                                    ],
                                )
                                
                                
                                if df.shape[0] > 2:
                                        df.to_csv(
                                            tracksavedir
                                            + '/'
                                            + 'Track'
                                            + str(self.ID)
                                            + 'tracklet'
                                            + str("_") + str(trackletid)
                                            + '.csv',
                                            index=False,
                                        )
                                
        
                                faketid = []
                                fakedist = []
                                for (tid, dist) in AllStartParent.items():
        
                                    if len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid', 'StartDistance'],
                                )
                                df.to_csv(
                                    tracksavedir + '/' + 'ParentFateStart' + '.csv', index=False
                                )
                                df
        
                                faketid = []
                                fakedist = []
                                for (tid, dist) in AllEndParent.items():
        
                                    if len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid', 'endDistance'],
                                )
                                df.to_csv(
                                    tracksavedir + '/' + 'ParentFateEnd' + '.csv', index=False
                                )
                                
        
                                faketid = []
                                fakedist = []
        
                                for (tid, dist) in AllStartChildren.items():
                                    if len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid + Trackletid', 'StartDistance'],
                                )
                                if df.shape[0] > 2:
                                        df.to_csv(
                                            tracksavedir + '/' + 'ChildrenFateStart' + '.csv',
                                            index=False,
                                        )
                                
        
                                faketid = []
                                fakedist = []
        
                                for (tid, dist) in AllEndChildren.items():
                                    if  len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid + Trackletid', 'EndDistance'],
                                )
                                if df.shape[0] > 2:
                                        df.to_csv(
                                            tracksavedir + '/' + 'ChildrenFateEnd' + '.csv', index=False
                                        )
                                
        
                            childrenstarts = AllStartChildren[
                                int(str(trackid) + str(trackletid))
                            ]
                            childrenends = AllEndChildren[int(str(trackid) + str("_") + str(trackletid))]
                            parentstarts = AllStartParent[trackid]
                            parentends = AllEndParent[trackid]
                            self.ax[0,0].plot(self.AllT, self.AllDistance)
                            self.ax[0,1].plot(parentstarts[1:], parentends[1:], 'og')
                            self.ax[0,1].plot(childrenstarts[1:], childrenends[1:], 'or')
                            self.figure.savefig(tracksavedir+ '/' + 'Track' + self.mode + str(self.ID) + '.png', dpi = 300)
                            self.figure.canvas.draw()
                            self.figure.canvas.flush_events()
        
        
        
                        TrackLayerTracklets[trackid].append(IDLocations)
        
        for (trackid, tracklets) in TrackLayerTracklets.items():

            
            locations = tracklets[1]
            
            df = pd.DataFrame(locations)
            #Create deltat, deltaz, deltay, deltax
            derivativedf = df.diff()
            t = df[0][1:]
            deltat = derivativedf[0][1:] 
            deltaz = derivativedf[1][1:] 
            deltay = derivativedf[2][1:] 
            deltax = derivativedf[3][1:] 
            
            print('Gaussfits for trackid: ', trackid)
            gmodel = Model(gaussian)
            meanz, stdz = norm.fit(deltaz) 
            countsz, binsz = np.histogram(deltaz,bins = self.nbins )
            binsz = binsz[:-1]
            try:
              GaussZ = gmodel.fit(countsz, x=binsz, amp=np.max(countsz), mu=meanz, std=stdz)
            except:
                GaussZ = None
            meany, stdy = norm.fit(deltay) 
            countsy, binsy = np.histogram(deltay, bins = self.nbins)
            binsy = binsy[:-1]
            try:
              GaussY = gmodel.fit(countsy, x=binsy, amp=np.max(countsy), mu=meany, std=stdy)
            except:
              GaussY = None  
                
            meanx, stdx = norm.fit(deltax) 
            countsx, binsx = np.histogram(deltax, bins = self.nbins)
            binsx = binsx[:-1]
            
            try:
               GaussX = gmodel.fit(countsx, x=binsx, amp=np.max(countsx), mu=meanx, std=stdx)
            except:
                GaussX = None
                
                
            self.ax[1,0].hist(binsz,binsz, weights = countsz)
            if GaussZ is not None:
                self.ax[1,0].plot(binsz,GaussZ.best_fit,'ro:',label='fit')
                self.ax[1,0].set_title('Gaussian Fit')
                self.ax[1,0].set_xlabel('Time')
                self.ax[1,0].set_ylabel('DisplacementZ')
            
            self.ax[1,1].hist(binsy,binsy, weights = countsy)
            
            if GaussY is not None:
                self.ax[1,1].plot(binsy,GaussY.best_fit,'ro:',label='fit')
                self.ax[1,1].set_title('Gaussian Fit')
                self.ax[1,1].set_xlabel('Time')
                self.ax[1,1].set_ylabel('DisplacementY')
            
            self.ax[1,2].hist(binsx,binsx, weights = countsx)
            
            if GaussX is not None:
                self.ax[1,2].plot(binsx,GaussX.best_fit,'ro:',label='fit')
                self.ax[1,2].set_title('Gaussian Fit')
                self.ax[1,2].set_xlabel('Time')
                self.ax[1,2].set_ylabel('DisplacementX')
            
            
            plt.tight_layout()
            plt.show()    
            
            if GaussZ is not None:
               print('Fit in Z', GaussZ.fit_report())
            if GaussY is not None:   
               print('Fit in Y', GaussY.fit_report())
            if GaussX is not None:     
              print('Fit in X', GaussX.fit_report())
        
        
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        
    def draw(self):
            TrackLayerTracklets = {}
            self.trackviewer.status = str(self.ID)
            for i in range(0, len(self.all_track_properties)):
                        trackid, alltracklets, DividingTrajectory = self.all_track_properties[i]
                        if self.ID is not None and self.ID == trackid:
                            TrackLayerTracklets = self.track(
                                TrackLayerTracklets, trackid, alltracklets
                            )
                        if self.ID == None or self.ID == 'all':
                            TrackLayerTracklets = self.track(
                                TrackLayerTracklets, trackid, alltracklets
                            )
    
            for (trackid, tracklets) in TrackLayerTracklets.items():
    
                tracklets = tracklets[1]
                properties = {
                'time': np.asarray(tracklets)[:, 0],
                
            }
                if len(tracklets) > 0:
                    self.trackviewer.add_tracks(
                        np.asarray(tracklets), name=self.tracklines + str(trackid), properties=properties
                    )
    
            

    def track(self, TrackLayerTracklets, trackid, alltracklets):

        TrackLayerTracklets[trackid] = [trackid]
        list_tracklets = []
        for (trackletid, tracklets) in alltracklets.items():

            Locationtracklets = tracklets[1]
            if len(Locationtracklets) > 0:
                Locationtracklets = sorted(
                    Locationtracklets, key=sortFirst, reverse=False
                )
                for tracklet in Locationtracklets:
                    (
                        t,
                        z,
                        y,
                        x,
                        total_intensity,
                        mean_intensity,
                        cellradius,
                        distance,
                        prob_inside,
                        trackletspeed,
                        trackletdirection,
                        DividingTrajectory,
                    ) = tracklet
                    if (
                        DividingTrajectory == self.DividingTrajectory
                        and self.ID == None
                        or self.ID == 'all'
                    ):
                        list_tracklets.append(
                            [
                                int(str(trackletid)),
                                int(float(t)/self.calibration[3]) ,
                                float(z)/self.calibration[2] ,
                                float(y)/self.calibration[1],
                                float(x)/self.calibration[0] ,
                            ]
                        )
                    else:
                        list_tracklets.append(
                            [
                                int(str(trackletid)),
                                int(float(t)/self.calibration[3]) ,
                                float(z)/self.calibration[2] ,
                                float(y)/self.calibration[1],
                                float(x)/self.calibration[0] ,
                            ]
                        )
                TrackLayerTracklets[trackid].append(list_tracklets)

        return TrackLayerTracklets

class AllTrackViewer(object):
    def __init__(
        self,
        originalviewer,
        Raw,
        Seg,
        Mask,
        savedir,
        calibration,
        all_track_properties,
        ID,
        canvas,
        ax,
        
        figure,
        DividingTrajectory = None,
        saveplot=False,
        
        window_size=3,
        mode='fate',
        deltat = 1
        
    ):

        self.trackviewer = originalviewer
        self.savedir = savedir
        Path(self.savedir).mkdir(exist_ok=True)
        self.Raw = Raw
        self.Seg = Seg
        self.Mask = Mask
        self.calibration = calibration
        self.mode = mode
        if ID == 'all':
            self.ID = ID
        elif ID is not None:
            self.ID = int(ID)
        else:
            self.ID = ID
        self.saveplot = saveplot
        self.canvas = canvas
        self.ax = ax
        self.deltat = deltat
        self.window_size = window_size
        self.figure = figure
        self.all_track_properties = all_track_properties
        self.DividingTrajectory = DividingTrajectory
        self.tracklines = 'Tracks'
        self.boxes = {}
        self.labels = {}
        for layer in list(self.trackviewer.layers):

            if self.tracklines in layer.name or layer.name in self.tracklines:
                self.trackviewer.layers.remove(layer)
        self.draw()
        if self.mode == 'fate':
            self.plot()
        if self.mode == 'intensity':
            self.plotintensity()
        
    def plot(self):

        for i in range(self.ax.shape[0]):
            self.ax[i].cla()

        self.ax[0].set_title("distance_to_boundary")
        self.ax[0].set_xlabel("minutes")
        self.ax[0].set_ylabel("um")

        self.ax[1].set_title("cell_tissue_localization")
        self.ax[1].set_xlabel("start_distance")
        self.ax[1].set_ylabel("end_distance")

        # Execute the function
        IDLocations = []
        TrackLayerTracklets = {}
        for i in tqdm(range(0, len(self.all_track_properties))):
                    trackid, alltracklets, DividingTrajectory = self.all_track_properties[i]
                    if self.ID == trackid or self.ID == 'all':
                        tracksavedir = self.savedir + '/' + 'TrackID' +  str(self.ID)
                        Path(tracksavedir).mkdir(exist_ok=True)
                        AllStartParent[trackid] = [trackid]
                        AllEndParent[trackid] = [trackid]
        
                        TrackLayerTracklets[trackid] = [trackid]
                        for (trackletid, tracklets) in alltracklets.items():
        
                            AllStartChildren[int(str(trackid) + str(trackletid))] = [
                                int(str(trackid) + str("_") +str(trackletid))
                            ]
                            AllEndChildren[int(str(trackid) + str(trackletid))] = [
                                int(str(trackid) +str("_") + str(trackletid))
                            ]
        
                            self.AllT = []
                            self.AllArea = []
                            self.AllIntensity = []
                            self.AllProbability = []
                            self.AllSpeed = []
                            self.AllDirection = []
                            self.AllSize = []
                            self.AllDistance = []
                            TrackLayerTrackletsList = []
        
                            Locationtracklets = tracklets[1]
                            if len(Locationtracklets) > 0:
                                Locationtracklets = sorted(
                                    Locationtracklets, key=sortFirst, reverse=False
                                )
        
                                for tracklet in Locationtracklets:
                                    (
                                        t,
                                        z,
                                        y,
                                        x,
                                        total_intensity,
                                        mean_intensity,
                                        cellradius,
                                        distance,
                                        prob_inside,
                                        speed,
                                        directionality,
                                        DividingTrajectory,
                                    ) = tracklet
                                    TrackLayerTrackletsList.append([trackletid, t, z, y, x])
                                    IDLocations.append([t, z, y, x])
                                    
                                    self.AllT.append(int(float(t )))
                                    self.AllSpeed.append("{:.1f}".format(float(speed)))
                                    self.AllDirection.append("{:.1f}".format(float(directionality)))
                                    self.AllProbability.append(
                                        "{:.2f}".format(float(prob_inside))
                                    )
                                    self.AllDistance.append("{:.1f}".format(float(distance)))
                                    self.AllSize.append("{:.1f}".format(float(cellradius)))
                                    if str(self.ID) + str(trackletid) not in AllID:
                                        AllID.append(str(self.ID) + str(trackletid))
                                if trackletid == 0:
                                    if len(self.AllDistance) > 1:
                                            AllStartParent[trackid].append(float(self.AllDistance[0]))
                                            AllEndParent[trackid].append(float(self.AllDistance[-1]))
        
                                else:
                                    if len(self.AllDistance) > 1:
                                            AllStartChildren[
                                                int(str(trackid) + str("_") +  str(trackletid))
                                            ].append(float(self.AllDistance[0]))
                                            AllEndChildren[int(str(trackid) +str("_") + str(trackletid))].append(
                                                float(self.AllDistance[-1])
                                            )
        
                            self.AllSpeed = MovingAverage(
                                self.AllSpeed, window_size=self.window_size
                            )
                            self.AllDirection = MovingAverage(
                                self.AllDirection, window_size=self.window_size
                            )
                            self.AllDistance = MovingAverage(
                                self.AllDistance, window_size=self.window_size
                            )
                            self.AllSize = MovingAverage(
                                self.AllSize, window_size=self.window_size
                            )
                            self.AllProbability = MovingAverage(
                                self.AllProbability, window_size=self.window_size
                            )
                            self.AllT = MovingAverage(self.AllT, window_size=self.window_size)
                            if self.saveplot == True:
                                SaveIds.append(self.ID)
                                
                                df = pd.DataFrame(
                                    list(
                                        zip(
                                            self.AllT,
                                            self.AllSize,
                                            self.AllDistance,
                                            self.AllProbability,
                                            self.AllSpeed,
                                            self.AllDirection
                                        )
                                    ),
                                    columns=[
                                        'Time',
                                        'Cell Size',
                                        'Distance to Border',
                                        'Inner Cell Probability',
                                        'Cell Speed',
                                        'Directionality'
                                    ],
                                )
                                
                                
                                if df.shape[0] > 2:
                                        df.to_csv(
                                            tracksavedir
                                            + '/'
                                            + 'Track'
                                            + str(self.ID)
                                            + 'tracklet'
                                            + str("_") + str(trackletid)
                                            + '.csv',
                                            index=False,
                                        )
                                
        
                                faketid = []
                                fakedist = []
                                for (tid, dist) in AllStartParent.items():
        
                                    if len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid', 'StartDistance'],
                                )
                                df.to_csv(
                                    tracksavedir + '/' + 'ParentFateStart' + '.csv', index=False
                                )
                                df
        
                                faketid = []
                                fakedist = []
                                for (tid, dist) in AllEndParent.items():
        
                                    if len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid', 'endDistance'],
                                )
                                df.to_csv(
                                    tracksavedir + '/' + 'ParentFateEnd' + '.csv', index=False
                                )
                                
        
                                faketid = []
                                fakedist = []
        
                                for (tid, dist) in AllStartChildren.items():
                                    if len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid + Trackletid', 'StartDistance'],
                                )
                                if df.shape[0] > 2:
                                        df.to_csv(
                                            tracksavedir + '/' + 'ChildrenFateStart' + '.csv',
                                            index=False,
                                        )
                                
        
                                faketid = []
                                fakedist = []
        
                                for (tid, dist) in AllEndChildren.items():
                                    if  len(dist) > 1:
                                        faketid.append(tid)
                                        fakedist.append(dist[1])
        
                                df = pd.DataFrame(
                                    list(zip(faketid, fakedist)),
                                    columns=['Trackid + Trackletid', 'EndDistance'],
                                )
                                if df.shape[0] > 2:
                                        df.to_csv(
                                            tracksavedir + '/' + 'ChildrenFateEnd' + '.csv', index=False
                                        )
                                
        
                            childrenstarts = AllStartChildren[
                                int(str(trackid) + str(trackletid))
                            ]
                            childrenends = AllEndChildren[int(str(trackid) + str("_") + str(trackletid))]
                            parentstarts = AllStartParent[trackid]
                            parentends = AllEndParent[trackid]
                            self.ax[0].plot(self.AllT, self.AllDistance)
                            self.ax[1].plot(parentstarts[1:], parentends[1:], 'og')
                            self.ax[1].plot(childrenstarts[1:], childrenends[1:], 'or')
                            self.figure.savefig(tracksavedir+ '/' + 'Track' + self.mode + str(self.ID) + '.png', dpi = 300)
                            self.figure.canvas.draw()
                            self.figure.canvas.flush_events()
        
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.SaveStats()

    
    def plotintensity(self):

        for i in range(self.ax.shape[0]):
            self.ax[i].cla()

        self.ax[0].set_title("intensity")
        self.ax[0].set_xlabel("minutes")
        self.ax[0].set_ylabel("arb_units")

        self.ax[1].set_title("fourier_transform")
        self.ax[1].set_xlabel("frequency")
        self.ax[1].set_ylabel("arb_units")

        # Execute the function

        IDLocations = []
        TrackLayerTracklets = {}
        for i in range(0, len(self.all_track_properties)):
            trackid, alltracklets, DividingTrajectory = self.all_track_properties[i]
            if self.ID == trackid:
                        AllStartParent[trackid] = [trackid]
                        AllEndParent[trackid] = [trackid]
        
                        TrackLayerTracklets[trackid] = [trackid]
                        for (trackletid, tracklets) in alltracklets.items():
        
                            self.AllT = []
                            self.AllIntensity = []
                            TrackLayerTrackletsList = []
        
                            max_int = 1
                            Locationtracklets = tracklets[1]
                            if len(Locationtracklets) > 0:
                                Locationtracklets = sorted(
                                    Locationtracklets, key=sortFirst, reverse=False
                                )
        
                                for tracklet in Locationtracklets:
                                    (
                                        t,
                                        z,
                                        y,
                                        x,
                                        total_intensity,
                                        mean_intensity,
                                        cellradius,
                                        distance,
                                        prob_inside,
                                        speed,
                                        direction,
                                        DividingTrajectory,
                                    ) = tracklet
                                    if float(total_intensity) > max_int:
                                        max_int = float(total_intensity)
                                    TrackLayerTrackletsList.append([trackletid, t, z, y, x])
                                    IDLocations.append([t, z, y, x])
                                    self.AllT.append(int(float(t)))
        
                                    self.AllIntensity.append(
                                        "{:.1f}".format((float(total_intensity)))
                                    )
                                    if str(self.ID) + str(trackletid) not in AllID:
                                        AllID.append(str(self.ID) + str(trackletid))
        
                            self.AllIntensity = MovingAverage(
                                self.AllIntensity, window_size=self.window_size
                            )
                            for i in range(0, len(self.AllIntensity)):
                                self.AllIntensity[i] = self.AllIntensity[i] / (max_int)
                            self.AllT = MovingAverage(self.AllT, window_size=self.window_size)
                            pointsample = len(self.AllIntensity)
                            if pointsample > 0:
                                xf = fftfreq(pointsample, self.calibration[3])
                                fftstrip = fft(self.AllIntensity)
                                ffttotal = np.abs(fftstrip)
                                xf = xf[0 : len(xf) // 2]
                                ffttotal = ffttotal[0 : len(ffttotal) // 2]
                                if self.saveplot == True:
                                    SaveIds.append(self.ID)
                                    tracksavedir = self.savedir + '/' + str(self.ID)
                                    Path(tracksavedir).mkdir(exist_ok=True)
                                    
                                    df = pd.DataFrame(
                                        list(zip(self.AllT, xf, ffttotal)),
                                        columns=['Time', 'Frequency', 'FFT'],
                                    )
                                    df.to_csv(
                                        tracksavedir
                                        + '/'
                                        + 'Track_Frequency'
                                        + str(self.ID)
                                        + 'tracklet'
                                        + str("_") +  str(trackletid)
                                        + '.csv',
                                        index=False,
                                    )
                                    
                                self.figure.savefig(tracksavedir+ '/' + 'Track' + self.mode + str(self.ID) + '.png', dpi = 300)    
                                self.ax[0].plot(self.AllT, self.AllIntensity)
                                self.ax[1].plot(xf, ffttotal)
                                self.ax[1].set_yscale('log')        
                                
                                self.figure.canvas.draw()
                                self.figure.canvas.flush_events()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.SaveStats()

    def draw(self):
        TrackLayerTracklets = {}
        self.trackviewer.status = str(self.ID)
        for i in range(0, len(self.all_track_properties)):
                    trackid, alltracklets, DividingTrajectory = self.all_track_properties[i]
                    if self.ID is not None and self.ID == trackid:
                        TrackLayerTracklets = self.track(
                            TrackLayerTracklets, trackid, alltracklets
                        )
                    if self.ID == None or self.ID == 'all':
                        TrackLayerTracklets = self.track(
                            TrackLayerTracklets, trackid, alltracklets
                        )

        for (trackid, tracklets) in TrackLayerTracklets.items():

            tracklets = tracklets[1]
            if len(tracklets) > 0:
                properties = {
                'time': np.asarray(tracklets)[:, 0],
                
            }
                self.trackviewer.add_tracks(
                    np.asarray(tracklets), name=self.tracklines + str(trackid), properties = properties
                )

       
        

    def track(self, TrackLayerTracklets, trackid, alltracklets):

        TrackLayerTracklets[trackid] = [trackid]
        list_tracklets = []
        for (trackletid, tracklets) in alltracklets.items():

            Locationtracklets = tracklets[1]
            if len(Locationtracklets) > 0:
                Locationtracklets = sorted(
                    Locationtracklets, key=sortFirst, reverse=False
                )
                for tracklet in Locationtracklets:
                    (
                        t,
                        z,
                        y,
                        x,
                        total_intensity,
                        mean_intensity,
                        cellradius,
                        distance,
                        prob_inside,
                        trackletspeed,
                        trackletdirection,
                        DividingTrajectory,
                    ) = tracklet
                    if (
                        DividingTrajectory == self.DividingTrajectory
                        and self.ID == None
                        or self.ID == 'all'
                    ):
                        list_tracklets.append(
                            [
                                int(str(trackletid)),
                                int(float(t)/self.calibration[3]) ,
                                float(z)/self.calibration[2] ,
                                float(y)/self.calibration[1],
                                float(x)/self.calibration[0] ,
                            ]
                        )
                    else:
                        list_tracklets.append(
                            [
                                int(str(trackletid)),
                                int(float(t)/self.calibration[3]) ,
                                float(z)/self.calibration[2] ,
                                float(y)/self.calibration[1],
                                float(x)/self.calibration[0] ,
                            ]
                        )
                TrackLayerTracklets[trackid].append(list_tracklets)

        return TrackLayerTracklets

    def SaveStats(self):
        
        tracksavedir = self.savedir + '/' + "SavedTracks"
        Path(tracksavedir).mkdir(exist_ok=True) 
        count = globalcount
        for i in range(0, len(self.all_track_properties)):
                    trackid, alltracklets, DividingTrajectory = self.all_track_properties[i]
                    for (trackletid, tracklets) in alltracklets.items():
                       for j in range(0, len(SaveIds)):
                          currentid = SaveIds[j]
                          if trackid == currentid:
                             
                                
                                for (tid, dist) in AllStartParent.items():
        
                                    if tid == trackid and  len(dist) > 1:
                                        if tid not in parentstartid:
                                                parentstartid.append(tid)
                                                parentstartdist.append(dist[1])
        
                                
                                for (tid, dist) in AllEndParent.items():
                                    if tid not in parentendid:
                                      if tid == trackid and len(dist) > 1:
                                        parentendid.append(tid)
                                        parentenddist.append(dist[1])
                                
                                
        
                                for (tid, dist) in AllStartChildren.items():
                                    if tid not in childrenstartid:
                                      if tid == int(str(trackid) + str("_") +  str(trackletid)) and len(dist) > 1:
                                          childrenstartid.append(tid)
                                          childrenstartdist.append(dist[1])
                                
                                      
        
                                for (tid, dist) in AllEndChildren.items():
                                    if tid not in childrenendid:
                                      if tid == int(str(trackid) + str("_") +  str(trackletid)) and  len(dist) > 1:
                                          childrenendid.append(tid)
                                          childrenenddist.append(dist[1])
                                
                              
                                
                                
                          
        if count == "0":
                dfps = pd.DataFrame(
                    list(zip(parentstartid, parentstartdist)),
                    columns=['Trackid', 'StartDistance'],
                )
                dfpe = pd.DataFrame(
                    list(zip(parentendid, parentenddist)),
                    columns=['Trackid', 'endDistance'],
                )
                dfcs = pd.DataFrame(
                    list(zip(childrenstartid, childrenstartdist)),
                    columns=['Trackid + Trackletid', 'StartDistance'],
                )
                dfce = pd.DataFrame(
                                        list(zip(childrenendid, childrenenddist)),
                                        columns=['Trackid + Trackletid', 'EndDistance'],
                                    )
        else:
            
            dfps = pd.DataFrame(
                    list(zip(parentstartid, parentstartdist)), header = False
                )
            dfpe = pd.DataFrame(
                list(zip(parentendid, parentenddist), header = False)
            )
            dfcs = pd.DataFrame(
                list(zip(childrenstartid, childrenstartdist)), header = False
            )
            dfce = pd.DataFrame(
                                    list(zip(childrenendid, childrenenddist)), header = False
                                )
                
        csvname =  tracksavedir + '/' + 'ChildrenFateStart' + '.csv'
        if os.path.exists(csvname):
 
              os.remove(csvname)                             
        dfcs.to_csv(
            csvname,
            index=False, mode='a'
        )
        csvname = tracksavedir + '/' + 'ParentFateStart' + '.csv'   
        if os.path.exists(csvname):
 
              os.remove(csvname)
        dfps.to_csv(
            csvname, index=False, mode='a'
        )   
        csvname = tracksavedir + '/' + 'ParentFateEnd' + '.csv'
        if os.path.exists(csvname):
 
              os.remove(csvname)
        dfpe.to_csv(
            csvname, index=False, mode='a'
        )
        csvname = tracksavedir + '/' + 'ChildrenFateEnd' + '.csv'
        if os.path.exists(csvname):
 
              os.remove(csvname)
        dfce.to_csv(csvname, index=False, mode='a'
                                )                                
                                             
        count = globalcount + "1"

def FourierTransform(numbers, tcalibration):

    pointsample = len(numbers)
    xf = fftfreq(pointsample, tcalibration)
    fftstrip = fft(numbers)
    ffttotal = np.abs(fftstrip)

    return ffttotal, xf


def MovingAverage(numbers, window_size=3):

    without_nans = np.zeros_like(numbers)
    numbers_series = pd.Series(numbers)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1 :]
    return without_nans


def TrackMateLiveTracks(
    Raw,
    Seg,
    Mask,
    savedir,
    calibration,
    all_track_properties,
    DividingTrajectory,
    mode='fate',
):

    Raw = imread(Raw)
    if Seg is not None:
       Seg = imread(Seg)
       Seg = Seg.astype('uint16')
    if Mask is not None:
        
        Mask = Mask.astype('uint16')


    viewer = napari.view_image(Raw, name='Image')
        
    if Seg is not None:
        viewer.add_labels(Seg, name='SegImage')

    if Mask is not None:
        Boundary = Mask.copy()
        Boundary = Boundary.astype('uint16')
        viewer.add_labels(Boundary, name='Mask')

    ID = []

    if DividingTrajectory:
        ID = DividingTrackIds
    else:
        ID = NonDividingTrackIds
    trackbox = QComboBox()
    trackbox.addItem(Boxname)

    tracksavebutton = QPushButton('Save Track')
    for i in range(0, len(ID)):
            trackbox.addItem(str(ID[i]))
    trackbox.addItem('all')

    figure = plt.figure(figsize=(4, 4))
    multiplot_widget = FigureCanvas(figure)
    ax = multiplot_widget.figure.subplots(1, 2)
    
    
    
    width = 400
    dock_widget = viewer.window.add_dock_widget(
        multiplot_widget, name="TrackStats", area='right'
    )
    multiplot_widget.figure.tight_layout()
    viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)

    T = Raw.shape[0]
    animation_widget = AnimationWidget(viewer, savedir, 0, T)
    viewer.window.add_dock_widget(animation_widget, area='right')
    viewer.update_console({'animation': animation_widget.animation})

    AllTrackViewer(
        viewer,
        Raw,
        Seg,
        Mask,
        savedir,
        calibration,
        all_track_properties,
        None,
        multiplot_widget,
        ax,
        figure,
        DividingTrajectory,
        saveplot = False,
        mode=mode,
    )
    trackbox.currentIndexChanged.connect(
        lambda trackid=trackbox: AllTrackViewer(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            DividingTrajectory,
            saveplot = False,
            mode=mode,
        )
    )
    tracksavebutton.clicked.connect(
        lambda trackid=tracksavebutton: AllTrackViewer(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            DividingTrajectory,
            saveplot = True,
            mode=mode,
        )
    )

    viewer.window.add_dock_widget(trackbox, name="TrackID", area='left')
    viewer.window.add_dock_widget(tracksavebutton, name="Save TrackID", area='left')
    




def TrackMateLiveTracksGauss(
    Raw,
    Seg,
    Mask,
    savedir,
    calibration,
    all_track_properties,
    DividingTrajectory,
    mode='fate',
    nbins = 10
):

    Raw = imread(Raw)
    if Seg is not None:
       Seg = imread(Seg)
       Seg = Seg.astype('uint16')
    if Mask is not None:
        
        Mask = Mask.astype('uint16')


    viewer = napari.view_image(Raw, name='Image')
        
    if Seg is not None:
        viewer.add_labels(Seg, name='SegImage')

    if Mask is not None:
        Boundary = Mask.copy()
        Boundary = Boundary.astype('uint16')
        viewer.add_labels(Boundary, name='Mask')

    ID = []

    if DividingTrajectory:
        ID = DividingTrackIds
    else:
        ID = NonDividingTrackIds
    trackbox = QComboBox()
    trackbox.addItem(Boxname)

    tracksavebutton = QPushButton('Save Track')
    for i in range(0, len(ID)):
            trackbox.addItem(str(ID[i]))
    trackbox.addItem('all')

    figure = plt.figure(figsize=(4, 4))
    multiplot_widget = FigureCanvas(figure)
    ax = multiplot_widget.figure.subplots(2, 3)
    
    
    
    width = 400
    dock_widget = viewer.window.add_dock_widget(
        multiplot_widget, name="TrackStats", area='right'
    )
    multiplot_widget.figure.tight_layout()
    viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)

    T = Raw.shape[0]
    animation_widget = AnimationWidget(viewer, savedir, 0, T)
    viewer.window.add_dock_widget(animation_widget, area='right')
    viewer.update_console({'animation': animation_widget.animation})

    AllTrackViewerGauss(
        viewer,
        Raw,
        Seg,
        Mask,
        savedir,
        calibration,
        all_track_properties,
        None,
        multiplot_widget,
        ax,
        figure,
        DividingTrajectory,
        saveplot = False,
        mode=mode,
        nbins = nbins
    )
    trackbox.currentIndexChanged.connect(
        lambda trackid=trackbox: AllTrackViewerGauss(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            DividingTrajectory,
            saveplot = False,
            mode=mode,
            nbins = nbins
        )
    )
    tracksavebutton.clicked.connect(
        lambda trackid=tracksavebutton: AllTrackViewerGauss(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            DividingTrajectory,
            saveplot = True,
            mode=mode,
            nbins = nbins
        )
    )

    viewer.window.add_dock_widget(trackbox, name="TrackID", area='left')
    viewer.window.add_dock_widget(tracksavebutton, name="Save TrackID", area='left')
    


def ShowAllTracksGauss(
    Raw,
    Seg,
    Mask,
    savedir,
    calibration,
    all_track_properties,
    mode='fate',
    nbins = 10
):

    print('Reading Image') 
    Raw = imread(Raw)
    if Seg is not None:
        Seg = imread(Seg)
        Seg = Seg.astype('uint16')
        print('Reading Seg Image')   
    if Mask is not None:
        
        Mask = Mask.astype('uint16')
        print('Reading Mask Image') 
        
        
    print('Building Napari in augenblick') 
    
    viewer = napari.Viewer()
    
    viewer.add_image(Raw, name='Image')
        
    if Seg is not None:
        viewer.add_labels(Seg, name='SegImage')

    if Mask is not None:
        Boundary = Mask.copy()
        Boundary = Boundary.astype('uint16')
        viewer.add_labels(Boundary, name='Mask')
        
        
    print('Building Napari GUI')
    ID = AllTrackIds
    trackbox = QComboBox()
    trackbox.addItem(Boxname)
    tracksavebutton = QPushButton('Save Track')
    for i in range(0, len(ID)):
            trackbox.addItem(str(ID[i]))
    trackbox.addItem('all')
    figure = plt.figure(figsize=(4, 4))
    multiplot_widget = FigureCanvas(figure)
    ax = multiplot_widget.figure.subplots(2, 3)
    width = 400
    dock_widget = viewer.window.add_dock_widget(
        multiplot_widget, name="TrackStats", area='right'
    )
    print('Adding widgets')
    multiplot_widget.figure.tight_layout()
    viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)
    T = Raw.shape[0]
    animation_widget = AnimationWidget(viewer, savedir, 0, T)
    viewer.window.add_dock_widget(animation_widget, area='right')
    viewer.update_console({'animation': animation_widget.animation})
     
    AllTrackViewerGauss(
        viewer,
        Raw,
        Seg,
        Mask,
        savedir,
        calibration,
        all_track_properties,
        None,
        multiplot_widget,
        ax,
        figure,
        None,
        saveplot = False,
        mode=mode,
        nbins = nbins
    )
    trackbox.currentIndexChanged.connect(
        lambda trackid=trackbox: AllTrackViewerGauss(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            None,
            saveplot = False,
            mode=mode,
            nbins = nbins
        )
    )
    
    tracksavebutton.clicked.connect(
        lambda trackid=tracksavebutton: AllTrackViewerGauss(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            None,
            saveplot = True,
            mode=mode,
            nbins = nbins
        )
    )
    
    print('About to open Napari')
    viewer.window.add_dock_widget(trackbox, name="TrackID", area='left')
    viewer.window.add_dock_widget(tracksavebutton, name="Save TrackID", area='left')
    napari.run() 
def ShowAllTracks(
    Raw,
    Seg,
    Mask,
    savedir,
    calibration,
    all_track_properties,
    mode='fate'
):

    print('Reading Image') 
    Raw = imread(Raw)
    if Seg is not None:
        Seg = imread(Seg)
        Seg = Seg.astype('uint16')
        print('Reading Seg Image')   
    if Mask is not None:
        
        Mask = Mask.astype('uint16')
        print('Reading Mask Image') 
        
        
    print('Building Napari in augenblick') 
    
    viewer = napari.Viewer()
    
    viewer.add_image(Raw, name='Image')
        
    if Seg is not None:
        viewer.add_labels(Seg, name='SegImage')

    if Mask is not None:
        Boundary = Mask.copy()
        Boundary = Boundary.astype('uint16')
        viewer.add_labels(Boundary, name='Mask')
        
        
    print('Building Napari GUI')
    ID = AllTrackIds
    trackbox = QComboBox()
    trackbox.addItem(Boxname)
    tracksavebutton = QPushButton('Save Track')
    for i in range(0, len(ID)):
            trackbox.addItem(str(ID[i]))
    trackbox.addItem('all')
    figure = plt.figure(figsize=(4, 4))
    multiplot_widget = FigureCanvas(figure)
    ax = multiplot_widget.figure.subplots(1, 2)
    width = 400
    dock_widget = viewer.window.add_dock_widget(
        multiplot_widget, name="TrackStats", area='right'
    )
    print('Adding widgets')
    multiplot_widget.figure.tight_layout()
    viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)
    T = Raw.shape[0]
    animation_widget = AnimationWidget(viewer, savedir, 0, T)
    viewer.window.add_dock_widget(animation_widget, area='right')
    viewer.update_console({'animation': animation_widget.animation})
    
    AllTrackViewer(
        viewer,
        Raw,
        Seg,
        Mask,
        savedir,
        calibration,
        all_track_properties,
        None,
        multiplot_widget,
        ax,
        figure,
        None,
        saveplot = False,
        mode=mode
    )
    trackbox.currentIndexChanged.connect(
        lambda trackid=trackbox: AllTrackViewer(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            None,
            saveplot = False,
            mode=mode
        )
    )
    
    tracksavebutton.clicked.connect(
        lambda trackid=tracksavebutton: AllTrackViewer(
            viewer,
            Raw,
            Seg,
            Mask,
            savedir,
            calibration,
            all_track_properties,
            trackbox.currentText(),
            multiplot_widget,
            ax,
            figure,
            None,
            saveplot = True,
            mode=mode
        )
    )
    
    print('About to open Napari')
    viewer.window.add_dock_widget(trackbox, name="TrackID", area='left')
    viewer.window.add_dock_widget(tracksavebutton, name="Save TrackID", area='left')
    napari.run()
    
    
def DistancePlotter():

    plt.plot(AllStartParent, AllEndParent, 'g')
    plt.title('Parent Start and End Distance')
    plt.xlabel('End Distance')
    plt.ylabel('Start Distance')
    plt.show()

    plt.plot(AllStartChildren, AllEndChildren, 'r')
    plt.title('Children Start and End Distance')
    plt.xlabel('End Distance')
    plt.ylabel('Start Distance')
    plt.show()


def on_click():
    return True


def sortTracks(List):

    return int(float(List[2]))


def sortFirst(List):

    return int(float(List[0]))

def gaussian(x, amp, mu, std):
    return amp * exp(-(x-mu)**2 / std)



def BiModalgaussian(x, ampA, muA, stdA, ampB, muB, stdB):
    return ampA * exp(-(x-muA)**2 / stdA) + ampB * exp(-(x-muB)**2 / stdB)

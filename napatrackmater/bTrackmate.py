import codecs
import csv
import math
import multiprocessing
import os
import xml.etree.cElementTree as et
from functools import partial
import multiprocessing
from pathlib import Path
import sys
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

'''Define function to run multiple processors and pool the results together'''





Boxname = 'TrackBox'
AttributeBoxname = 'AttributeIDBox'
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

def prob_sigmoid(x):
    return 1 - math.exp(-x)


def CreateTrackCheckpoint(ImageName, LabelName, MaskName, Name, savedir):

    Mask = None

    Label = imread(LabelName)

    Image = imread(ImageName)

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
            indices = np.transpose(np.asarray(indices))
            real_indices = indices.copy()
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

        Boundary = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])
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
                indices = np.transpose(np.asarray(indices))
                real_indices = indices.copy()
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

        indices = np.transpose(np.asarray(indices))
        real_indices = indices.copy()
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
                    Root = target_id
                    if Root == Leaf:
                        break
                else:
                    break
        non_dividing_tracklets.append([trackletid, tracklet, trackletspeed])
    return non_dividing_tracklets


def analyze_dividing_tracklets(root_leaf, split_points, spot_object_source_target):

    dividing_tracklets = []
    # Make tracklets
    Root = root_leaf[0]

    visited = []
    # For the root we need to go forward
    tracklet = []
    trackletspeed = []
    tracklet.append(Root)
    trackletspeed.append(0)
    trackletid = 0
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

    dividing_tracklets.append([trackletid, tracklet, trackletspeed])

    trackletid = 1

    # Exclude the split point near root
    for i in range(0, len(split_points) - 1):
        Start = split_points[i]
        tracklet = []
        trackletspeed = []
        tracklet.append(Start)
        trackletspeed.append(0)
        Othersplit_points = split_points.copy()
        Othersplit_points.pop(i)
        while Start is not Root:
            for (
                source_id,
                target_id,
                edge_time,
                directional_rate_change,
                speed,
            ) in spot_object_source_target:

                if Start == target_id:

                    Start = source_id
                    if Start in visited:
                        break
                    tracklet.append(source_id)
                    visited.append(source_id)
                    trackletspeed.append(speed)
                    if Start in Othersplit_points:
                        break

        dividing_tracklets.append([trackletid, tracklet, trackletspeed])
        trackletid = trackletid + 1

    for i in range(0, len(root_leaf)):
        leaf = root_leaf[i]
        # For leaf we need to go backward
        tracklet = []
        trackletspeed = []
        tracklet.append(leaf)
        trackletspeed.append(0)
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
                    visited.append(source_id)
                    tracklet.append(source_id)
                    trackletspeed.append(speed)
        dividing_tracklets.append([trackletid, tracklet, trackletspeed])
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
        trackletid, tracklets, trackletspeeds = alltracklets[i]
        location_prop_dist[trackletid] = [trackletid]
        for k in range(0, len(tracklets)):

            tracklet = tracklets[k]
            trackletspeed = trackletspeeds[k]
            cell_source_id = tracklet
            frame, z, y, x = Uniqueobjects[int(cell_source_id)]
         
            total_intensity, mean_intensity, real_time, cellradius, pixel_volume = Uniqueproperties[
                int(cell_source_id)
            ]

            if Mask is not None:

                testlocation = (z, y, x)

                tree, indices, masklabel, masklabelvolume = TimedMask[str(int(frame))]

                region_label = Mask[
                    int(float(frame) / calibration[3]),
                    int(float(z) / calibration[2]),
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
                    DividingTrajectory
                ]
            )

        location_prop_dist[trackletid].append(current_location_prop_dist)

    return location_prop_dist

def relabel_track_property(
    tracks,
    filtered_track_ids,
    Uniqueobjects,
    Segimage
):

    location_prop_dist = {}
    
    for track in tracks.findall('Track'):

        track_id = int(track.get("TRACK_ID"))
        
        spot_object_source_target = []
        if track_id in filtered_track_ids:
            location_prop_dist[track_id] = [track_id]
            
            for edge in track.findall('Edge'):
            
                source_id = edge.get('SPOT_SOURCE_ID')
                target_id = edge.get('SPOT_TARGET_ID')
                edge_time = edge.get('EDGE_TIME')
               
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
            split_points, root_leaf = Multiplicity(spot_object_source_target)

            # Determine if a track has divisions or none
            if len(split_points) > 0:
                split_points = split_points[::-1]
                DividingTrajectory = True
            else:
                DividingTrajectory = False
          
            if DividingTrajectory == True:
                DividingTrackIds.append(track_id)
                AllTrackIds.append(track_id)
                alltracklets = analyze_dividing_tracklets(
                    root_leaf, split_points, spot_object_source_target
                )

            if DividingTrajectory == False:
                NonDividingTrackIds.append(track_id)
                AllTrackIds.append(track_id)
                alltracklets = analyze_non_dividing_tracklets(
                    root_leaf, spot_object_source_target
                )
            
            for i in range(0, len(alltracklets)):
               current_location_prop_dist = []
               trackletid, tracklets, trackletspeeds = alltracklets[i]
               for k in range(0, len(tracklets)):
                        
                    tracklet = tracklets[k]
                    trackletspeed = trackletspeeds[k]
                    cell_source_id = tracklet
                    frame, z, y, x = Uniqueobjects[int(cell_source_id)]
                    current_location_prop_dist.append([int(float(frame)), int(float(z)), int(float(y)), int(float(x))])
        location_prop_dist[track_id].append(current_location_prop_dist)
    Segimage = RelabelCells(Segimage,location_prop_dist)

    return Segimage

def import_TM_XML_Relabel(xml_path, Segimage,spot_csv, track_csv, savedir):
    
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
    AllValues = []
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
        try:
          if k != 'TRACK_ID':  
             
             AllValues.append(track_dataset[k].astype('float'))
             AllKeys.append(k)
        except:
            pass
    
   
    
    Viz = VizCorrect(Segimage, Name, savedir,AllKeys, AllValues)
    Viz.showNapari()
    
    #Alllocations = [LocationT.tolist(),LocationZ.tolist(),LocationY.tolist(),LocationX.tolist()]
    #Allproperties = [Track_id.tolist(), ]

   
            # for each tracklet get real_time,z,y,x,total_intensity, mean_intensity, cellradius, distance, prob_inside
    #NewSegimage = RelabelCells(Segimage,Alllocations)
           
    #return  NewSegimage


def normalizeZeroOne(x, scale = 255 * 255):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x * scale
 
def RelabelCells(Segimage,Alllocations, AllKeys, AllValues):
    
        NewSegimage = np.zeros(Segimage.shape)
        assert len(AllKeys) == len(AllValues)
        
        
        
        for (k,v) in location_prop_dist.items():
            
                indices = v[1:][0]
                for i in range(0, Segimage.shape[0]):
                   Labelimage = Segimage[i,:]
                   NewLabelimage = NewSegimage[i,:]
                   props =  measure.regionprops(Labelimage,Labelimage) 
                   centroids = [prop.centroid for prop in props] 
                   labels = [prop.label for prop in props]
                   tree = spatial.cKDTree(centroids)
                   Labels = []
                   
                   for index in indices:  
                          
                          time = index[0]
                          z = index[1]
                          y = index[2]
                          x = index[3]  
                            
                          if i == time:
                                    
                                    if z < Labelimage.shape[0]:
                                       pt = (z,y,x)       
                                       closest =  tree.query(pt)
                                       indexlist = centroids.index((centroids[closest[1]][0], centroids[closest[1]][1], centroids[closest[1]][2]))
                                    
                                       Labels.append(labels[indexlist])
                                       
                                            
                   for regionlabel in Labels:
                      try:
                         NewLabelimage[np.where(Labelimage == regionlabel) ] = k
                      except:
                        pass
                   NewSegimage[i,:] = NewLabelimage
        return NewSegimage   

class VizCorrect(object):

        def __init__(self, Segimage, Name, savedir,AllKeys, AllValues ):
            
            
               self.Segimage = Segimage
               self.Name = Name
               self.savedir = savedir
               self.AllKeys = AllKeys
               self.AllValues = AllValues
               Path(self.savedir).mkdir(exist_ok=True)
               
               
            
        def showNapari(self):
                 
                 self.viewer = napari.Viewer()
                 self.viewer.theme = 'dark'
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
                 
                 for attributename in self.AllKeys:
                     Attributeids.append(attributename)
                     
                 
                
                    
                 Attributeidbox = QComboBox()   
                 Attributeidbox.addItem(AttributeBoxname)   
                
                 savebutton = QPushButton(' Save Relabelled Image')
                 computebutton = QPushButton(' compute Relabelled Image')
                 for i in range(0, len(Attributeids)):
                     
                     
                     Attributeidbox.addItem(str(Attributeids[i]))
                     

                 
                 Attributeidbox.currentIndexChanged.connect(
                 lambda trackid = Attributeidbox: self.second_image_add(
                         
                         Attributeidbox.currentText(),
                         
                         self.Name,
                         False,
                         False
                    
                )
            )            
                 
                 savebutton.clicked.connect(
                 lambda trackid = Attributeidbox: self.second_image_add(
                         
                         Attributeidbox.currentText(),
                         
                         self.Name,
                         False,
                         True
                    
                )
            ) 
                 computebutton.clicked.connect(
                 lambda trackid = Attributeidbox: self.second_image_add(
                         
                         Attributeidbox.currentText(),
                         
                         self.Name,
                         True,
                         False
                    
                )
            )   
                    
                
                 
                 self.viewer.window.add_dock_widget(Attributeidbox, name="Color Attributes", area='right') 
                 self.viewer.window.add_dock_widget(computebutton, name="Compute Relabelled Image", area='left') 
                 self.viewer.window.add_dock_widget(savebutton, name="Save Relabelled Image", area='left') 
                 
                
        
                 
                        
        

                                     
        def second_image_add(self, attribute, imagename,compute = False, save = False):
                
                       
                        
                        if compute:
                          
                          self.boxes = {}
                          self.labels = {}
                          print("Computing region boxes")
                          for i in tqdm(range(0, self.Segimage.shape[0])):
                              timeboxes = []
                              timelabels = []
                              ThreeDimage = self.Segimage[i,:]
                              
                              for region in regionprops(ThreeDimage):
                              
                                    timeboxes.append(region.bbox)
                                    timelabels.append(region.label)
                              self.boxes[i] = [i]     
                              self.boxes[i].append(timeboxes)
                              
                              self.labels[i] = [i]     
                              self.labels[i].append(timelabels)
                              
                          
                          for k in range(len(self.AllKeys)):
                            
                            locations = []
                            attrs = []
                            if self.AllKeys[k] ==  attribute:
                                
                                for attr, time, z, y, x in tqdm(zip(self.AllValues[k],self.AllValues[self.keyT],self.AllValues[self.keyZ],self.AllValues[self.keyY],self.AllValues[self.keyX] ), total = len(self.AllValues[k])):
                                       centroid = (time, z, y, x)
                                       locations.append([attr, centroid]) 
                                       
                                NewSegimage = self.Relabel(self.Segimage.copy(), locations)
                                self.viewer.add_labels(NewSegimage, name = self.Name + attribute)  
                             

                        if save:


                                ModifiedArraySeg = self.viewer.layers[self.Name + attribute].data 
                                ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
        
                                imwrite((self.savedir  +   self.Name + attribute+ '.tif' ) , ModifiedArraySeg)
        
                                                          
        def Conditioncheck(self, centroid, boxA, p, ndim):
            
              condition = False
            
              if centroid[p] >=  boxA[p]  and centroid[p] <=  boxA[p + ndim]:
                  
                   condition = True
                   
              return condition     
         
        def iou(self, boxA, centroid, label, relabelval):
            
            ndim = len(centroid)
            inside = False
            
            Condition = [self.Conditioncheck(centroid, boxA, p, ndim) for p in range(0,ndim)]
                
            inside = all(Condition)
            
            if inside:
                
                return boxA, relabelval 
            
            else:
                
                return boxA, label
                    
        def Relabel(self, image, locations):
        
               print("Relabelling image with chosen trackmate attribute")
               NewSegimage = image.copy()
               for p in tqdm(range(0, NewSegimage.shape[0])):
                   
                   sliceimage = NewSegimage[p,:]
                   originallabels = []
                   newlabels = []
                   for  relabelval, centroid in locations:
                       
                        time, z, y, x = centroid
                   
                   
                        if p == time: 
                               
                               timeboxes = self.boxes[time][1]
                               timelabels = self.labels[time][1]
                               for i  in range(len(timeboxes)):
                                  box =  timeboxes[i]
                                  originallabel = timelabels[i]
                                  box, returnval = self.iou(box, (z,y,x), originallabel, relabelval)
                                                
                                  if math.isnan(returnval):
                                      returnval = -1
                                  if abs(returnval - originallabel) > 0:
                                      originallabels.append(int(originallabel))
                                      newlabels.append(int(returnval))
                                     
                  
                      
                   relabeled = map_array(sliceimage, np.asarray(originallabels), np.asarray(newlabels))
                   NewSegimage[p,:] = relabeled
                               
               return NewSegimage                     
               
              
                    
            

def import_TM_XML(xml_path, image, Segimage = None, Mask=None):
    
    image = daskread(image)[0,:]
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
                Spotobject.get('POSITION_T'),
                Radius,
                QUALITY
            ]

    all_track_properties = []
    for track in tracks.findall('Track'):

        track_id = int(track.get("TRACK_ID"))

        spot_object_source_target = []
        if track_id in filtered_track_ids:
            for edge in track.findall('Edge'):

                source_id = edge.get('SPOT_SOURCE_ID')
                target_id = edge.get('SPOT_TARGET_ID')
                edge_time = edge.get('EDGE_TIME')
               
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
            split_points, root_leaf = Multiplicity(spot_object_source_target)

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
                Segimage,
                TimedMask,
                [xcalibration, ycalibration, zcalibration, tcalibration],
                DividingTrajectory
            )
            all_track_properties.append([track_id, location_prop_dist, DividingTrajectory])
   
    return all_track_properties, Boundary, [
        xcalibration,
        ycalibration,
        zcalibration,
        tcalibration,
    ]

 

def Multiplicity(spot_object_source_target):

    split_points = []
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
        scount = 0

    return split_points, root_leaf






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
        self.window_size = window_size
        self.figure = figure
        self.all_track_properties = all_track_properties
        self.DividingTrajectory = DividingTrajectory
        self.tracklines = 'Tracks'

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
        for i in range(0, len(self.all_track_properties)):
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
                                        DividingTrajectory,
                                    ) = tracklet
                                    TrackLayerTrackletsList.append([trackletid, t, z, y, x])
                                    IDLocations.append([t, z, y, x])
                                    self.AllT.append(int(float(t * self.calibration[3])))
                                    self.AllSpeed.append("{:.1f}".format(float(speed)))
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
                                        )
                                    ),
                                    columns=[
                                        'Time',
                                        'Cell Size',
                                        'Distance to Border',
                                        'Inner Cell Probability',
                                        'Cell Speed',
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
                            TrackLayerTracklets = []
        
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
                                        DividingTrajectory,
                                    ) = tracklet
                                    if float(total_intensity) > max_int:
                                        max_int = float(total_intensity)
                                    TrackLayerTracklets.append([trackletid, t, z, y, x])
                                    IDLocations.append([t, z, y, x])
                                    self.AllT.append(int(float(t * self.calibration[3])))
        
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
                                        + str("_") +  str(_trackletid)
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
                self.trackviewer.add_tracks(
                    np.asarray(tracklets), name=self.tracklines + str(trackid), colormap = 'twilight', tail_length = sys.float_info.max
                )

        self.trackviewer.theme = 'light'
        self.trackviewer.dims.ndisplay = 3

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
                                int(float(t)) / self.calibration[3],
                                float(z) / self.calibration[2],
                                float(y) / self.calibration[1],
                                float(x) / self.calibration[0],
                            ]
                        )
                    else:
                        list_tracklets.append(
                            [
                                int(str(trackletid)),
                                int(float(t)) / self.calibration[3],
                                float(z) / self.calibration[2],
                                float(y) / self.calibration[1],
                                float(x) / self.calibration[0],
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
    napari.run()

def ShowAllTracks(
    Raw,
    Seg,
    Mask,
    savedir,
    calibration,
    all_track_properties,
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
            None,
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
            None,
            saveplot = True,
            mode=mode,
        )
    )
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

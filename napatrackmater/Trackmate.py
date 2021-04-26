import xml.etree.cElementTree as et
import os
import numpy as np
import pandas as pd
import csv
from skimage import measure
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider, QComboBox, QPushButton
from tqdm import tqdm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math
import matplotlib.pyplot as plt
from btrack.dataio import  _PyTrackObjectFactory
from btrack.dataio import export_CSV
from btrack.dataio import export_LBEP
import btrack
from skimage.measure import label
from skimage.filters import sobel
from btrack.constants import BayesianUpdates
from tifffile import imread, imwrite
from btrack.dataio import import_CSV
from skimage.segmentation import find_boundaries
from PyQt5.QtCore import pyqtSlot
from scipy import spatial 
import pandas as pd
from .napari_animation import AnimationWidget
Boxname = 'TrackBox'
pd.options.display.float_format = '${:,.2f}'.format


ParentDistances = {}
ChildrenDistances = {}

AllStartParent = []
AllEndParent = []
AllID = []
AllStartChildren = []
AllEndChildren = []

def Velocity(Source, Target, xycalibration, zcalibration, tcalibration):
    
    
    ts,zs,ys,xs = Source
    
    tt,zt,yt,xt = Target
    
  
    
    Velocity = (float(zs)* zcalibration - float(zt)* zcalibration) * (float(zs)* zcalibration - float(zt)* zcalibration) + (float(ys)* xycalibration - float(yt)* xycalibration) * (float(ys)* xycalibration - float(yt)* xycalibration) + (float(xs)* xycalibration - float(xt)* xycalibration) * (float(xs)* xycalibration - float(xt)* xycalibration)
    
    return math.sqrt(Velocity)/ max((float(tt)* tcalibration-float(ts)* tcalibration),1)
    


def GetBorderMask(Mask):
    
    ndim = len(Mask.shape)
    #YX shaped object
    if ndim == 2:
        Mask = label(Mask)
        Boundary = find_boundaries(Mask)
        
    #TYX shaped object    
    if ndim == 3:
        
        Boundary = np.zeros([Mask.shape[0], Mask.shape[1], Mask.shape[2]])
        for i in range(0, Mask.shape[0]):
            
            Mask[i,:] = label(Mask[i,:])   
            Boundary[i,:] = find_boundaries(Mask[i,:])
            
            
        #TZYX shaped object        
    if ndim == 4:

        Boundary = np.zeros([Mask.shape[0], Mask.shape[1], Mask.shape[2], Mask.shape[3]])
        
        #Loop over time
        for i in range(0, Mask.shape[0]):
            
            Mask[i,:] = label(Mask[i,:])   
            
            for j in range(0,Mask.shape[1]):
               
                          Boundary[i,j,:,:] = find_boundaries(Mask[i,j,:,:])    
        
    return Boundary  

     
        

"""
Convert an integer image into boundary points for 2,3 and 4D data

"""


def boundary_points(mask, xcalibration, ycalibration, zcalibration):
    
    ndim = len(mask.shape)
    
    timed_mask = {}
    #YX shaped object
    if ndim == 2:
        mask = label(mask)
        labels = []
        volumelabel = [] 
        tree = []
        properties = measure.regionprops(mask, mask)
        for prop in properties:
            
            labelimage = prop.image
            regionlabel = prop.label
            sizey = abs(prop.bbox[0] - prop.bbox[2]) * xcalibration
            sizex = abs(prop.bbox[1] - prop.bbox[3]) * ycalibration
            volume = sizey * sizex
            radius = math.sqrt(volume/math.pi)
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
                volumelabel.append(radius) 
        #This object contains list of all the points for all the labels in the Mask image with the label id and volume of each label    
        timed_mask[str(0)] = [tree, indices, labels, volumelabel]
        
        
    #TYX shaped object    
    if ndim == 3:
        
        Boundary = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])
        for i in range(0, mask.shape[0]):
            
            mask[i,:] = label(mask[i,:])
            properties = measure.regionprops(mask[i,:], mask[i,:])
            labels = []
            volumelabel = [] 
            tree = []
            for prop in properties:
                
                labelimage = prop.image
                regionlabel = prop.label
                sizey = abs(prop.bbox[0] - prop.bbox[2]) * ycalibration
                sizex = abs(prop.bbox[1] - prop.bbox[3]) * xcalibration
                volume = sizey * sizex
                radius = math.sqrt(volume/math.pi)
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
                        volumelabel.append(radius)
                
            timed_mask[str(i)] = [tree, indices, labels, volumelabel]
            
            
    #TZYX shaped object        
    if ndim == 4:

        Boundary = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]])
        
        #Loop over time
        for i in range(0, mask.shape[0]):
            
            mask[i,:] = label(mask[i,:])
            properties = measure.regionprops(mask[i,:], mask[i,:])
            labels = []
            volumelabel = []
            tree = []
            for prop in properties:
                
                labelimage = prop.image
                regionlabel = prop.label
                sizez = abs(prop.bbox[0] - prop.bbox[3])* zcalibration
                sizey = abs(prop.bbox[1] - prop.bbox[4])* ycalibration
                sizex = abs(prop.bbox[2] - prop.bbox[5])* xcalibration
                volume = sizex * sizey * sizez 
                radius = math.pow(3 * volume / ( 4 * math.pi), 1.0/3.0)
                #Loop over Z
                if regionlabel > 1: 
                    for j in range(int(prop.bbox[0]),int(prop.bbox[3])):
               
                          Boundary[i,j,:labelimage.shape[1],:labelimage.shape[2]] = find_boundaries(labelimage[j,:,:])
                else:
                    for j in range(int(prop.bbox[0]),int(prop.bbox[3])):
               
                          Boundary[i,j,:,:] = find_boundaries(mask[i,j,:,:])
                
                indices = np.where(Boundary[i,:] > 0)
                
                indices = np.transpose(np.asarray(indices))
                real_indices = indices.copy()
                for j in range(0, len(real_indices)):
                    
                    real_indices[j][0] = real_indices[j][0] * zcalibration
                    real_indices[j][1] = real_indices[j][1] * ycalibration
                    real_indices[j][2] = real_indices[j][2] * xcalibration
                    
                
                tree.append(spatial.cKDTree(real_indices))
                if regionlabel not in labels:
                    labels.append(regionlabel)
                    volumelabel.append(radius) 
                
            
            
            timed_mask[str(i)] = [tree, indices, labels, volumelabel]    

    return timed_mask
    






'''
In this method we purge the short tracklets to exclude them for further tracking process
'''
def PurgeTracklets(root_leaf, split_points, spot_object_source_target, DividingTrajectory, mintracklength = 2):
    
    #root_leaf object contains root in the begining and leafes after that, we remove the split point ID and leaf ID the corresponds to short tracklets

    Curatedroot_leaf = []
    Curatedsplit_points = split_points.copy()
    Root = root_leaf[0]
    Curatedroot_leaf = root_leaf.copy()
    if DividingTrajectory == True:
        Visited = []
        for i in range(1, len(root_leaf)):
                                leaf = root_leaf[i]
                                tracklength = 0
                                while(leaf not in split_points and leaf != Root):
                                    for source_id, target_id, edge_time in spot_object_source_target:
                                        # Search for the target id corresponding to leaf                        
                                        if leaf == target_id:
                                              # Include the split points here
                                              #Once we find the leaf we move a step back to its source to find its source
                                              Leaf = source_id
                                              tracklength = tracklength + 1
                                              RootSplitPoint = Leaf
                                              if Leaf in split_points:
                                                  break
                                              if Leaf in Visited:
                                                break
                                              Visited.append(source_id)
                                if tracklength < mintracklength:  
                                    try:
                                       Curatedsplit_points.remove(RootSplitPoint)
                                       Curatedroot_leaf.remove(Leaf)
                                    except:
                                        pass
                                    
                             
    return Curatedsplit_points, Curatedroot_leaf                                          



def analyze_non_dividing_tracklets(root_leaf, spot_object_source_target):
    
      non_dividing_tracklets = []
      if len(root_leaf) > 0:
                             Root = root_leaf[0]
                             Leaf = root_leaf[-1]
                             tracklet = []
                             trackletid = 0
                             tracklet.append(Root)
                             #For non dividing trajectories iterate from Root to the only Leaf
                             while(Root != Leaf):
                                        for source_id,target_id, edge_time, directional_rate_change, speed in spot_object_source_target:
                                                if Root == source_id:
                                                      tracklet.append([source_id, directional_rate_change, speed])
                                                      Root = target_id
                                                      if Root==Leaf:
                                                          break
                                                else:
                                                    break
                             non_dividing_tracklets.append([trackletid, tracklet]) 
                             sorted_non_dividing_tracklets = tracklet_sorter(non_dividing_tracklets, spot_object_source_target)
                             
      return sorted_non_dividing_tracklets                       

def analyze_dividing_tracklets(root_leaf, split_points, spot_object_source_target):
    
    
                            dividing_tracklets = []
                            print("Analyzing Dividing Trajectory")
                            #Make tracklets
                            Root = root_leaf[0]
                            
                            visited = []
                            #For the root we need to go forward
                            tracklet = []
                            tracklet.append(Root)
                            trackletid = 0
                            RootCopy = Root
                            visited.append(Root)
                            while(RootCopy not in split_points and RootCopy not in root_leaf[1:]):
                                for source_id,target_id, edge_time, directional_rate_change, speed in spot_object_source_target:
                                        # Search for the target id corresponding to leaf                        
                                        if RootCopy == source_id:
                                              
                                              #Once we find the leaf we move a step fwd to its target to find its target
                                              RootCopy = target_id
                                              if RootCopy in split_points:
                                                  break
                                              if RootCopy in visited:
                                                break
                                              visited.append(target_id)
                                              tracklet.append([source_id, directional_rate_change, speed])
                                              
                            dividing_tracklets.append([trackletid, tracklet])
                            
                            trackletid = 1       
                            for i in range(1, len(root_leaf)):
                                leaf = root_leaf[i]
                                #For leaf we need to go backward
                                tracklet = []
                                tracklet.append(leaf)
                                while(leaf not in split_points and leaf != Root):
                                    for source_id,target_id, edge_time, directional_rate_change, speed in spot_object_source_target:
                                        # Search for the target id corresponding to leaf                        
                                        if leaf == target_id:
                                              # Include the split points here
                                              
                                              #Once we find the leaf we move a step back to its source to find its source
                                              leaf = source_id
                                              if leaf in split_points:
                                                  break
                                              if leaf in visited:
                                                break
                                              visited.append(source_id)
                                              tracklet.append([source_id, directional_rate_change, speed])
                                dividing_tracklets.append([trackletid, tracklet]) 
                                trackletid = trackletid + 1
                            
                            
                            # Exclude the split point near root    
                            for i in range(0, len(split_points) -1):
                                Start = split_points[i]
                                tracklet = []
                                tracklet.append(Start)
                                Othersplit_points = split_points.copy()
                                Othersplit_points.pop(i)
                                while(Start is not Root):
                                    for source_id,target_id, edge_time, directional_rate_change, speed in spot_object_source_target:
                                        
                                        if Start == target_id:
                                            
                                            Start = source_id
                                            if Start in visited:
                                                break
                                            tracklet.append([source_id, directional_rate_change, speed])
                                            visited.append(source_id)
                                            if Start in Othersplit_points:
                                                break
                                            
                                dividing_tracklets.append([trackletid, tracklet]) 
                                trackletid = trackletid + 1
                            
                            sorted_dividing_tracklets = tracklet_sorter(dividing_tracklets, spot_object_source_target)    
                            
                            return sorted_dividing_tracklets    
                        
                        
def tracklet_properties(tracklet_ids, dict_ordered_tracklets, Uniqueobjects, Uniqueproperties, Mask, TimedMask):
    
    
                            location_prop_dist = {}
                            for idxs in tracklet_ids:
                                location_prop_dist[idxs] = [idxs]
                                current_location_prop_dist = []
                                tracklets = dict_ordered_tracklets[idxs]
                                for cell_source_id, directional_rate_change, speed in tracklets:
                                    
                                      frame,z,y,x = Uniqueobjects[int(cell_source_id)]
                                      total_intensity, mean_intensity, real_time, cellradius =  Uniqueproperties[int(cell_source_id)]
                                      
                                      if Mask is not None:
                                                                
                                                                testlocation = (z,y,x)
                                                                tree, indices, masklabel, masklabelvolume = TimedMask[str(int(frame))]
                                                               
                                                                region_label = Mask[int(frame), int(z), int(y) , int(x)] 
                                                                
                                                                for k in range(0, len(masklabel)):
                                                                    currentlabel = masklabel[k]
                                                                    currentvolume = masklabelvolume[k]
                                                                    currenttree = tree[k]
                                                                    #Get the location and distance to the nearest boundary point
                                                                    distance, location = currenttree.query(testlocation)
                                                                    distance = max(0,distance  - cellradius)
                                                                    if currentlabel == region_label and region_label > 0:
                                                                            prob_inside = max(0,(distance - cellradius) / currentvolume)
                                                                    else:
                                                                        
                                                                            prob_inside = 0 
                                      else:
                                                                distance = 0
                                                                prob_inside = 0
                                                                
                                      current_location_prop_dist.append([real_time,z,y,x,total_intensity, mean_intensity, cellradius, distance, prob_inside])      
                        
                                location_prop_dist[idxs].append(current_location_prop_dist)
                            
                            return location_prop_dist    

def import_TM_XML(xml_path, Segimage, image = None, Mask = None, mintracklength = 2):
    
        Name = os.path.basename(os.path.splitext(xml_path)[0])
        savedir = os.path.dirname(xml_path)
        root = et.fromstring(open(xml_path).read())
          
        filtered_track_ids = [int(track.get('TRACK_ID')) for track in root.find('Model').find('FilteredTracks').findall('TrackID')]
        
        #Extract the tracks from xml
        tracks = root.find('Model').find('AllTracks')
        settings = root.find('Settings').find('ImageData')
        
        #Extract the cell objects from xml
        Spotobjects = root.find('Model').find('AllSpots') 
        
        #Make a dictionary of the unique cell objects with their properties        
        Uniqueobjects = {}
        Uniqueproperties = {}
        
        xcalibration = settings.get('pixelwidth')
        ycalibration = settings.get('pixelheight')
        zcalibration = settings.get('voxeldepth')
        if Mask is not None:
            if len(Mask.shape) < len(Segimage.shape):
                # T Z Y X
                UpdateMask = np.zeros([Segimage.shape[0], Segimage.shape[1], Segimage.shape[2], Segimage.shape[3]])
                for i in range(0, UpdateMask.shape[0]):
                    for j in range(0, UpdateMask.shape[1]):
                        
                        UpdateMask[i,j,:,:] = Mask[i,:,:]
            else:
                UpdateMask = Mask
            Mask = UpdateMask.astype('uint16')
            TimedMask = boundary_points(Mask, xcalibration, ycalibration, zcalibration)
        
        for frame in Spotobjects.findall('SpotsInFrame'):
            
            for Spotobject in frame.findall('Spot'):
                #Create object with unique cell ID
                cell_id = int(Spotobject.get("ID"))
                #Get the TZYX location of the cells in that frame
                Uniqueobjects[cell_id] = [Spotobject.get('FRAME'),Spotobject.get('POSITION_Z'), Spotobject.get('POSITION_Y'), Spotobject.get('POSITION_X') ]
                #Get other properties associated with the Spotobject
                Uniqueproperties[cell_id] = [Spotobject.get('TOTAL_INTENSITY_CH1')
                                                ,Spotobject.get('MEAN_INTENSITY_CH1'), Spotobject.get('POSITION_T'), Spotobject.get('RADIUS')]
                
        dividing_tracks = []
        non_dividing_tracks = []
        for track in tracks.findall('Track'):

            track_id = int(track.get("TRACK_ID"))
            spot_object_source_target = []
            if track_id in filtered_track_ids:
                print('Creating Tracklets of TrackID', track_id)
                for edge in track.findall('Edge'):
                   
                   source_id = edge.get('SPOT_SOURCE_ID')
                   target_id = edge.get('SPOT_TARGET_ID')
                   edge_time = edge.get('EDGE_TIME')
                   directional_rate_change = edge.get('DIRECTIONAL_CHANGE_RATE')
                   speed = edge.get('SPEED')
                   
                   
                   spot_object_source_target.append([source_id,target_id, edge_time, directional_rate_change, speed])
                
                #Sort the tracks by edge time  
                spot_object_source_target = sorted(spot_object_source_target, key = sortTracks , reverse = False)
                
                # Get all the IDs, uniquesource, targets attached, leaf, root, splitpoint IDs
                split_points, root_leaf = Multiplicity(spot_object_source_target)
                
                
                # Determine if a track has divisions or none
                if len(split_points) > 0:
                    split_points = split_points[::-1]
                    DividingTrajectory = True
                else:
                    DividingTrajectory = False
                    
                # Remove dangling tracklets, done in BTrackmate also    
                split_points, root_leaf = PurgeTracklets(root_leaf, split_points, spot_object_source_target, DividingTrajectory, mintracklength = mintracklength)     
                    
                tstart = 0    
                for source_id, target_id, edge_time in spot_object_source_target:
                     if root_leaf[0] == source_id:    
                             Source = Uniqueobjects[int(source_id)]
                             tstart = int(float(Source[0]))
                             break
                    
                
                if DividingTrajectory == True:
                    
                            tracklet_ids, dict_ordered_tracklets = analyze_dividing_tracklets(root_leaf, split_points, spot_object_source_target)
                                                
                            location_prop_dist = tracklet_properties(tracklet_ids, dict_ordered_tracklets, Uniqueobjects, Uniqueproperties, Mask, TimedMask)
                            
                            
                            dict_dividing_trackobjects, dict_dividing_speedobjects, dividing_trackobjects, dividing_tracklet_objects = TrackobjectCreator(sorted_dividing_tracklets, Uniqueobjects, xycalibration, zcalibration, tcalibration)

                            dividing_tracks.append([track_id,dict_dividing_trackobjects, dict_dividing_speedobjects, dividing_trackobjects, dividing_tracklet_objects, sorted_dividing_tracklets, tstart])

                            dividing_tracks = sorted(dividing_tracks, key = sortID, reverse = False)
                            
                if DividingTrajectory == False:
                    
                            tracklet_ids, dict_ordered_tracklets = analyze_non_dividing_tracklets(root_leaf, spot_object_source_target)    
                            
                            location_prop_dist = tracklet_properties(tracklet_ids, dict_ordered_tracklets, Uniqueobjects, Uniqueproperties, Mask, TimedMask)
                             
                            dict_non_dividing_trackobjects, dict_non_dividing_speedobjects, non_dividing_trackobjects, non_dividing_tracklet_objects = TrackobjectCreator(sorted_non_dividing_tracklets, Uniqueobjects, xycalibration, zcalibration, tcalibration)

                            non_dividing_tracks.append([track_id,dict_non_dividing_trackobjects, dict_non_dividing_speedobjects, non_dividing_trackobjects, non_dividing_tracklet_objects, sorted_non_dividing_tracklets, tstart])

                            non_dividing_tracks = sorted(non_dividing_tracks, key = sortID, reverse = False)   
                            
                            
        # Write all tracks to csv file as ID, T, Z, Y, X
        ID = []
        start_id = {}
        
        RegionID = {}
        VolumeID = {}
        locationID = {}    

        Tloc = []
        Zloc = []
        Yloc = []
        Xloc = []
        
        for trackid, dict_dividing_trackobjects, dict_dividing_speedobjects, dividing_trackobjects, dividing_tracklet_objects, sorted_dividing_tracklets, tstart in dividing_tracks:
            
             print('Computing Tracklets for TrackID:', trackid)  
             RegionID[trackid] = [trackid]
             VolumeID[trackid] = [trackid]
             locationID[trackid] = [trackid]
             start_id[trackid] = [trackid]
             tracklet_region_id = {}
             tracklet_volume_id = {}
             tracklet_location_id = {}
             
             start_id[trackid].append(tstart)
             

             Speedloc = []
             DistanceBoundary = []
             ProbabilityInside = []
             SlocZ = []
             SlocY = []
             SlocX = []
             Vloc = []
             Iloc = []
             for j in tqdm(range(0, len(dividing_tracklet_objects))):
                         
                                    Spottrackletid = tracklet_objects[j]
                                    tracklet_region_id[Spottrackletid] = [Spottrackletid]
                                    tracklet_volume_id[Spottrackletid] = [Spottrackletid]
                                    tracklet_location_id[Spottrackletid] = [Spottrackletid]
                                    TrackletLocation = []
                                    TrackletRegion = []
                                    TrackletVolume = []
                                    
                                    DictSpotobject = dict_track_objects[Spottrackletid][1]
                                    DictVelocitySpotobject = dict_speed_objects[Spottrackletid][1]
                                    
                                    for i in range(0, len(DictSpotobject)): 
                                           
                                           Spotobject = DictSpotobject[i]
                                           VelocitySpotobject = DictVelocitySpotobject[i]
                                           t = int(float(Spotobject[0]))
                                           z = int(float(Spotobject[1]))
                                           y = int(float(Spotobject[2]))
                                           x = int(float(Spotobject[3]))
                                           
                                          
                                           speed = (float(VelocitySpotobject))
                                           ID.append(trackid)
                                           Tloc.append(t)
                                           Zloc.append(z)
                                           Yloc.append(y)
                                           Xloc.append(x)
                                           Speedloc.append(speed)
                                           if t < Segimage.shape[0]:
                                                   CurrentSegimage = Segimage[t,:]
                                                   if image is not None:
                                                           Currentimage = image[t,:]
                                                           properties = measure.regionprops(CurrentSegimage, Currentimage)
                                                   if image is None:
                                                           properties = measure.regionprops(CurrentSegimage, CurrentSegimage)
                                                           
                                                   TwoDCoordinates = [(prop.centroid[1] , prop.centroid[2]) for prop in properties]
                                                   TwoDtree = spatial.cKDTree(TwoDCoordinates)
                                                   TwoDLocation = (y ,x)
                                                   closestpoint = TwoDtree.query(TwoDLocation)
                                                   for prop in properties:
                                                       
                                                       
                                                       if int(prop.centroid[1]) == int(TwoDCoordinates[closestpoint[1]][0]) and int(prop.centroid[2]) == int(TwoDCoordinates[closestpoint[1]][1]):
                                                           
                                                           
                                                            sizeZ = abs(prop.bbox[0] - prop.bbox[3]) * zcalibration
                                                            sizeY = abs(prop.bbox[1] - prop.bbox[4]) * xycalibration
                                                            sizeX = abs(prop.bbox[2] - prop.bbox[5]) * xycalibration
                                                            Area = prop.area
                                                            intensity = np.sum(prop.image)
                                                            Vloc.append(Area)
                                                            SlocZ.append(sizeZ)
                                                            SlocY.append(sizeY)
                                                            SlocX.append(sizeX)
                                                            Iloc.append(intensity)
                                                            TrackletRegion.append([1,sizeZ, sizeY,sizeX])
                                                            
                                                            
                                                            
                                                            # Compute distance to the boundary
                                                            if Mask is not None:
                                                                
                                                                testlocation = (z * zcalibration ,y * xycalibration ,x * xycalibration)
                                                                tree, indices, masklabel, masklabelvolume = TimedMask[str(int(t))]
                                                                
                                                                
                                                                cellradius = math.sqrt( sizeX * sizeX + sizeY * sizeY)/4
                                                               
                                                                Regionlabel = Mask[int(t), int(z), int(y) , int(x)] 
                                                                for k in range(0, len(masklabel)):
                                                                    currentlabel = masklabel[k]
                                                                    currentvolume = masklabelvolume[k]
                                                                    currenttree = tree[k]
                                                                    #Get the location and distance to the nearest boundary point
                                                                    distance, location = currenttree.query(testlocation)
                                                                    distance = max(0,distance  - cellradius)
                                                                    if currentlabel == Regionlabel and Regionlabel > 0:
                                                                            probabilityInside = max(0,(distance) / currentvolume)
                                                                    else:
                                                                        
                                                                            probabilityInside = 0 
                                                            else:
                                                                distance = 0
                                                                probabilityInside = 0
                                                            
                                                            DistanceBoundary.append(distance)
                                                            ProbabilityInside.append(probabilityInside)
                                                            TrackletVolume.append([Area, intensity, speed, distance , probabilityInside])
                                                            TrackletLocation.append([t, z, y, x])
                           
                                           tracklet_location_id[Spottrackletid].append(TrackletLocation)
                                           tracklet_volume_id[Spottrackletid].append(TrackletVolume)
                                           tracklet_region_id[Spottrackletid].append(TrackletRegion)
                           
             locationID[trackid].append(tracklet_location_id)
             RegionID[trackid].append(tracklet_region_id)
             VolumeID[trackid].append(tracklet_volume_id)
                 
        df = pd.DataFrame(list(zip(ID,Tloc,Zloc,Yloc,Xloc, DistanceBoundary, ProbabilityInside, SlocZ, SlocY, SlocX, Vloc, Iloc, Speedloc)), index = None, 
                                              columns =['ID', 't', 'z', 'y', 'x', 'distBoundary', 'probInside', 'sizeZ', 'sizeY', 'sizeX', 'volume', 'intensity', 'speed'])

        df.to_csv(savedir + '/' + 'Extra' + Name +  '.csv')  
        df     
        
        # create the final data array: track_id, T, Z, Y, X
        
        df = pd.DataFrame(list(zip(ID,Tloc,Zloc,Yloc,Xloc)), index = None, 
                                              columns =['ID', 't', 'z', 'y', 'x'])

        df.to_csv(savedir + '/' + 'TrackMate' +  Name +  '.csv', index = False)  
        df

        return RegionID, VolumeID, locationID, Tracks, ID, start_id
    
 
def TrackobjectCreator(ordered_tracklets, Uniqueobjects, xycalibration, zcalibration, tcalibration):

                dict_track_objects = {}
                dict_speed_objects = {} 
                tracklet_objects = []
                for k in range(0, len(ordered_tracklets)):
                    
                        trackletid, tracklet = ordered_tracklets[k]
                        tracklet_objects.append(trackletid)
                        Trackobjects = []
                        Speedobjects = []
                        dict_track_objects[trackletid] = [trackletid]
                        dict_speed_objects[trackletid] = [trackletid]
                        for i in range(0, len(tracklet)):
                            source_id, timeID = tracklet[i]
                            if i < len(tracklet) - 1:
                              target_id, targettimeID = tracklet[i+1]
                            else:
                              target_id = source_id
                            #All tracks
                            Source = Uniqueobjects[int(source_id)][1]
                            Target = Uniqueobjects[int(target_id)][1]
                            speed = Velocity(Source, Target, xycalibration, zcalibration, tcalibration)
                            if Target not in Trackobjects:
                               Trackobjects.append(Target)
                            Speedobjects.append(speed)
                        dict_track_objects[trackletid].append(Trackobjects)    
                        dict_speed_objects[trackletid].append(Speedobjects)    
                return dict_track_objects, dict_speed_objects, Trackobjects, tracklet_objects
            
            
def tracklet_sorter(Tracklets, spot_object_source_target):

    
     ordered_tracklets = []  
     tracklet_ids = []
     dict_ordered_tracklets = {}
     for trackletid , tracklet in Tracklets:
         tracklet_ids.append(trackletid)
         time_tracklet = []
         visited = []
         dict_ordered_tracklets[trackletid] = [trackletid]
         for cellsource_id in tracklet:
             
              for source_id,target_id, edge_time, directional_rate_change, speed  in spot_object_source_target:
                  
                  if cellsource_id == source_id or cellsource_id == target_id:
                                if cellsource_id not in visited:          
                                   time_tracklet.append([ [cellsource_id, directional_rate_change, speed], edge_time])
                                   visited.append(cellsource_id)
                                   
         otracklet = sorted(time_tracklet, key = sortTracklet, reverse = False)
         dict_ordered_tracklets[trackletid].append(otracklet)
         if len(otracklet) > 0:
            ordered_tracklets.append([trackletid,otracklet])
     
     return tracklet_ids, dict_ordered_tracklets

    
def Multiplicity(spot_object_source_target):

     split_points = []
     root_leaf = []
     sources = []
     targets = []
     scount = 0
     
     for source_id,target_id, sourcetime, directional_rate_change, speed in spot_object_source_target:
         
         sources.append(source_id)
         targets.append(target_id)
     
     root_leaf.append(sources[0])   
     for source_id,target_id, sourcetime, directional_rate_change, speed in spot_object_source_target:
                    
                if target_id not in sources:
                    
                        root_leaf.append(target_id)
         
                for sec_source_id, sec_target_id, sec_sourcetime, sec_directional_rate_change, sec_speed in spot_object_source_target:             
                             if source_id == sec_source_id:
                                  scount = scount + 1
                if scount > 1:
                    split_points.append(source_id)
                scount = 0    
                
                
     return split_points, root_leaf

         

           
 
           
    
class TrackViewer(object):
    
    
    def __init__(self, originalviewer, Raw, Seg, Mask, locationID, RegionID, VolumeID,  scale, ID, start_id, canvas, ax, figure, savedir, saveplot, tcalibration):
        
        
        self.trackviewer = originalviewer
        self.Raw = Raw
        self.Seg = Seg
        self.Mask = Mask
        self.locationID = locationID
        self.RegionID = RegionID
        self.VolumeID = VolumeID
        self.scale = scale
        self.ID = ID
        self.start_id = start_id
        self.tcalibration = tcalibration
        self.saveplot = saveplot
        self.savedir = savedir
        self.layername = 'Trackpoints'
        self.layernamedot = 'Trackdot'
        self.tracklines = 'Tracklets'
        #Side plots
        self.canvas = canvas
        self.figure = figure
        self.ax = ax 
        self.AllLocations = {}
        self.AllRegions = {}

        self.LocationTracklets = []
        self.plot() 
    def plot(self):
        
            for i in range(self.ax.shape[0]):
                 for j in range(self.ax.shape[1]):
                                   self.ax[i,j].cla()
            if self.ID!=Boxname:
                
                        self.ax[0,0].set_title("CellSize")
                        self.ax[0,0].set_xlabel("minutes")
                        self.ax[0,0].set_ylabel("um")
                        
                        self.ax[1,0].set_title("Distance to Boundary")
                        self.ax[1,0].set_xlabel("minutes")
                        self.ax[1,0].set_ylabel("um")
                        
                        self.ax[0,1].set_title("Expectation Inner cell")
                        self.ax[0,1].set_xlabel("minutes")
                        self.ax[0,1].set_ylabel("Probability")
                        
                        self.ax[1,1].set_title("CellVelocity")
                        self.ax[1,1].set_xlabel("minutes")
                        self.ax[1,1].set_ylabel("um")
                        
                        self.ax[0,2].set_title("CellIntensity")
                        self.ax[0,2].set_xlabel("minutes")
                        self.ax[0,2].set_ylabel("Arb. units")
                        
                        self.ax[1,2].set_title("CellFate")
                        self.ax[1,2].set_xlabel("Start Distance")
                        self.ax[1,2].set_ylabel("End Distance")
                        
                        #Execute the function    
                        
                        Location = self.locationID[int(float(self.ID))][1]
                        Volume =  self.VolumeID[int(float(self.ID))][1]
                        Region =  self.RegionID[int(float(self.ID))][1]
                        self.AllLocations[self.ID] = [self.ID]
                        self.AllRegions[self.ID] = [self.ID]
                        
                        
                        IDLocations = []
                        IDRegions = []
                        for (trackletid, tracklet) in Location.items():
                            
                            #print('Trackletid', trackletid)
                            self.AllT = []
                            self.AllIntensity = []
                            self.AllArea = []
                            self.AllSpeed = []
                            self.AllSize = []
                            self.AllDistance = []
                            self.AllProbability = []
                            Volumetracklet = Volume[trackletid][1]
                            Regiontracklet = Region[trackletid][1]
                            Locationtracklet = tracklet[1]
                            TrackLayerTracklets = []
                            #print('Locationtracklet', Locationtracklet)
                            for i in range(0, len(Locationtracklet)):
                                        t, z, y, x = Locationtracklet[i]
                                        TrackLayerTracklets.append([trackletid, t, z, y, x])
                                        area, intensity, speed, distance, probability = Volumetracklet[i]
                                        #print('Track ID:', self.ID, trackletid, 'Timepoint', t)
                                        
                                        sizeT, sizeZ, sizeY, sizeX = Regiontracklet[i]
                                        
                                        IDLocations.append([t,z,y,x])
                                        IDRegions.append([sizeT, sizeZ, sizeY, sizeX])
                                        
                                        self.AllT.append(t * self.tcalibration)
                                        self.AllArea.append(area)
                                        self.AllIntensity.append(intensity)
                                        self.AllSpeed.append(speed)
                                        self.AllDistance.append(distance)
                                        self.AllProbability.append(probability)
                                        self.AllSize.append(math.sqrt(sizeY * sizeY + sizeX * sizeX)/4)
                                 
                            if str(self.ID) + str(trackletid) not in AllID:
                                      AllID.append(str(self.ID) + str(trackletid))
                                      if trackletid == 0: 
                                          AllStartParent.append(self.AllDistance[0])
                                          AllEndParent.append(self.AllDistance[-1])
                                          
                                      else:
                                          AllStartChildren.append(self.AllDistance[0])
                                          AllEndChildren.append(self.AllDistance[-1])
                                       
                                
                            self.ax[0,0].plot(self.AllT, self.AllSize)
                            self.ax[1,0].plot(self.AllT, self.AllDistance)
                            self.ax[0,1].plot(self.AllT, self.AllProbability)
                            self.ax[1,1].plot(self.AllT, self.AllSpeed)
                            self.ax[0,2].plot(self.AllT, self.AllIntensity)
                            self.ax[1,2].plot(AllStartParent, AllEndParent, 'og')
                            self.ax[1,2].plot(AllStartChildren, AllEndChildren, 'or')
                            self.LocationTracklets.append(TrackLayerTracklets)
                            if self.saveplot:
                                    df = pd.DataFrame(list(zip(self.AllT,self.AllSize,self.AllDistance,self.AllProbability,self.AllSpeed,self.AllIntensity)),  
                                                      columns =['Time', 'Cell Size', 'Distance to Border', 'Inner Cell Probability', 'Cell Speed', 'Cell Intensity'])
                                    df.to_csv(self.savedir + '/' + 'Track' +  str(self.ID) + 'tracklet' + str(trackletid) +  '.csv',index = False)  
                                    df
                                    
                                    df = pd.DataFrame(list(zip(AllStartParent,AllEndParent)),  
                                                      columns =['StartDistance', 'EndDistance'])
                                    df.to_csv(self.savedir + '/'  + 'ParentFate'  +  '.csv',index = False)  
                                    df
                                    
                                    df = pd.DataFrame(list(zip(AllStartChildren,AllEndChildren)),  
                                                      columns =['StartDistance', 'EndDistance'])
                                    df.to_csv(self.savedir + '/' + 'ChildrenFate'  +  '.csv',index = False)  
                                    df
                                    
                                    
                                    
                                    
                            
                        self.AllLocations[self.ID].append(IDLocations)
                        self.AllRegions[self.ID].append(IDRegions)
            self.canvas.draw()            
            self.UpdateTrack()   
            
            
    def SaveFig(self):
        
        if self.saveplot:
           self.figure.savefig(self.savedir + '/' + 'Track' +  str(self.ID) +  '.png', transparent = True )
           
           
            
                    
    def UpdateTrack(self):
        
        
        
                if self.ID != Boxname:
                    
                        
                    
                        for layer in list(self.trackviewer.layers):
                           
                                 if self.layername == layer.name:
                                     self.trackviewer.layers.remove(layer)
                                 if self.layernamedot == layer.name:
                                     self.trackviewer.layers.remove(layer)
                                     
                                 if self.tracklines in layer.name or layer.name in self.tracklines:
                                     self.trackviewer.layers.remove(layer)
        
                
                        tstart = self.start_id[int(float(self.ID))][1]
                        self.trackviewer.dims.set_point(0, tstart)
                        self.trackviewer.status = str(self.ID)
                        for i in range(0, len(self.LocationTracklets)):
                            self.trackviewer.add_tracks(np.asarray(self.LocationTracklets[i]), scale = self.scale, name= self.tracklines + str(i))
                            
                            
                        self.trackviewer.theme = 'light'
                        self.trackviewer.dims.ndisplay = 3
                        self.SaveFig()
                       
                        T = self.Seg.shape[0]
                        animation_widget = AnimationWidget(self.trackviewer, self.savedir,self.ID, T)
                        self.trackviewer.window.add_dock_widget(animation_widget, area='right')
                        self.trackviewer.update_console({'animation': animation_widget.animation})
                    
            

                
def TrackMateLiveTracks(Raw, Seg, Mask, savedir, scale, locationID, RegionID, VolumeID, ID, start_id, tcalibration):

    if Mask is not None and len(Mask.shape) < len(Seg.shape):
        # T Z Y X
        UpdateMask = np.zeros_like(Seg)
        for i in range(0, UpdateMask.shape[0]):
            for j in range(0, UpdateMask.shape[1]):
                
                UpdateMask[i,j,:,:] = Mask[i,:,:]
    else:
        UpdateMask = Mask
    
    Boundary = GetBorderMask(UpdateMask.copy())
    
    with napari.gui_qt():
            if Raw is not None:
                          
                          viewer = napari.view_image(Raw, scale = scale, name='Image')
                          Labels = viewer.add_labels(Seg, scale = scale, name = 'SegImage')
            else:
                          viewer = napari.view_image(Seg, scale = scale, name='SegImage')
                          
            if Mask is not None:
                
                          LabelsMask = viewer.add_labels(Boundary, scale = scale, name='Mask')
            
            trackbox = QComboBox()
            trackbox.addItem(Boxname)
            
            tracksavebutton = QPushButton('Save Track')
            saveplot = tracksavebutton.clicked.connect(on_click)
            
        
            for i in range(0, len(ID)):
                trackbox.addItem(str(ID[i]))
            try:
               figure = plt.figure(figsize = (5, 5))    
               multiplot_widget = FigureCanvas(figure)
               ax = multiplot_widget.figure.subplots(2,3)
            except:
                pass
            viewer.window.add_dock_widget(multiplot_widget, name = "TrackStats", area = 'right')
            multiplot_widget.figure.tight_layout()
            trackbox.currentIndexChanged.connect(lambda trackid = trackbox : TrackViewer(viewer, Raw, Seg, Mask, locationID, RegionID,
                                                                                         VolumeID, scale, trackbox.currentText(), start_id,multiplot_widget, ax, figure, savedir, saveplot = False, tcalibration = tcalibration))
            
            if saveplot:
                tracksavebutton.clicked.connect(lambda trackid = tracksavebutton : TrackViewer(viewer, Raw, Seg, Mask, locationID, RegionID,
                                                                                         VolumeID, scale, trackbox.currentText(), start_id,multiplot_widget, ax, figure, savedir, True, tcalibration))
                
            viewer.window.add_dock_widget(trackbox, name = "TrackID", area = 'left')
            viewer.window.add_dock_widget(tracksavebutton, name = "Save TrackID", area = 'left')
  
    
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
  
@pyqtSlot()            
def on_click():
        
         
         return True         
            
def sortTracks(List):
    
    return int(float(List[2]))

def sortID(List):
    
    return int(float(List[0]))


def sortTracklet(List):
    
    return int(float(List[1]))

def sortX(List):
    
    return int(float(List[-1]))

def sortY(List):
    
    return int(float(List[-2]))
    



from tqdm import tqdm
import numpy as np 
import codecs
import xml.etree.ElementTree as et
import pandas as pd
import math
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy import spatial
import dask as da
from typing import List
from scipy.fftpack import fft, fftfreq, fftshift, ifft
import os
class TrackMate(object):
    
    def __init__(self, xml_path, spot_csv_path, track_csv_path, edges_csv_path, AttributeBoxname, TrackAttributeBoxname, TrackidBox, channel_seg_image = None, image = None, mask = None):
        
        
        self.xml_path = xml_path
        self.spot_csv_path = spot_csv_path
        self.track_csv_path = track_csv_path 
        self.edges_csv_path = edges_csv_path
        self.image = image 
        self.mask = mask 
        self.channel_seg_image = channel_seg_image
        self.AttributeBoxname = AttributeBoxname
        self.TrackAttributeBoxname = TrackAttributeBoxname
        self.TrackidBox = TrackidBox
        self.spot_dataset, self.spot_dataset_index = get_csv_data(self.spot_csv_path)
        self.track_dataset, self.track_dataset_index = get_csv_data(self.track_csv_path)
        self.edges_dataset, self.edges_dataset_index = get_csv_data(self.edges_csv_path)
       
                                                        
        self.track_analysis_spot_keys = dict(
                spot_id="ID",
                track_id="TRACK_ID",
                quality="QUALITY",
                posix="POSITION_X",
                posiy="POSITION_Y",
                posiz="POSITION_Z",
                posit="POSITION_T",
                frame="FRAME",
                radius="RADIUS",
                mean_intensity_ch1="MEAN_INTENSITY_CH1",
                total_intensity_ch1="TOTAL_INTENSITY_CH1",
                mean_intensity_ch2="MEAN_INTENSITY_CH2",
                total_intensity_ch2="TOTAL_INTENSITY_CH2",
            )
        self.track_analysis_edges_keys = dict(
                spot_source_id="SPOT_SOURCE_ID",
                spot_target_id="SPOT_TARGET_ID",
                directional_change_rate="DIRECTIONAL_CHANGE_RATE",
                speed="SPEED",
                displacement="DISPLACEMENT",
                edge_time="EDGE_TIME",
                edge_x_location="EDGE_X_LOCATION",
                edge_y_location="EDGE_Y_LOCATION",
                edge_z_location="EDGE_Z_LOCATION",
            )
        self.track_analysis_track_keys = dict(
                number_spots="NUMBER_SPOTS",
                number_gaps="NUMBER_GAPS",
                number_splits="NUMBER_SPLITS",
                number_merges="NUMBER_MERGES",
                track_duration="TRACK_DURATION",
                track_start="TRACK_START",
                track_stop="TRACK_STOP",
                track_displacement="TRACK_DISPLACEMENT",
                track_x_location="TRACK_X_LOCATION",
                track_y_location="TRACK_Y_LOCATION",
                track_z_location="TRACK_Z_LOCATION",
                track_mean_speed="TRACK_MEAN_SPEED",
                track_max_speed="TRACK_MAX_SPEED",
                track_min_speed="TRACK_MIN_SPEED",
                track_median_speed="TRACK_MEDIAN_SPEED",
                track_std_speed="TRACK_STD_SPEED",
                track_mean_quality="TRACK_MEAN_QUALITY",
                total_track_distance="TOTAL_DISTANCE_TRAVELED",
                max_track_distance="MAX_DISTANCE_TRAVELED",
                mean_straight_line_speed="MEAN_STRAIGHT_LINE_SPEED",
                linearity_forward_progression="LINEARITY_OF_FORWARD_PROGRESSION",
                mean_directional_change_rate="MEAN_DIRECTIONAL_CHANGE_RATE",
            )

        self.frameid_key = self.track_analysis_spot_keys["frame"]
        self.zposid_key = self.track_analysis_spot_keys["posiz"]
        self.yposid_key = self.track_analysis_spot_keys["posiy"]
        self.xposid_key = self.track_analysis_spot_keys["posix"]
        self.spotid_key = self.track_analysis_spot_keys["spot_id"]
        self.trackid_key = self.track_analysis_spot_keys["track_id"]
        self.radius_key = self.track_analysis_spot_keys["radius"]
        self.quality_key = self.track_analysis_spot_keys["quality"]

        self.generationid_key = 'generation_id'
        self.trackletid_key = 'tracklet_id'
        self.uniqueid_key = 'unique_id'
        self.afterid_key = 'after_id'
        self.beforeid_key = 'before_id'
        self.dividing_key = 'dividing_normal'
        self.distance_cell_mask_key = 'distance_cell_mask'
        self.cellid_key = 'cell_id'
        self.acceleration_key = 'acceleration'

        self.mean_intensity_ch1_key = self.track_analysis_spot_keys["mean_intensity_ch1"]
        self.mean_intensity_ch2_key = self.track_analysis_spot_keys["mean_intensity_ch2"]
        self.total_intensity_ch1_key = self.track_analysis_spot_keys["total_intensity_ch1"]
        self.total_intensity_ch2_key = self.track_analysis_spot_keys["total_intensity_ch2"]

        self.spot_source_id_key = self.track_analysis_edges_keys["spot_source_id"]
        self.spot_target_id_key = self.track_analysis_edges_keys["spot_target_id"]
        self.directional_change_rate_key = self.track_analysis_edges_keys["directional_change_rate"] 
        self.speed_key = self.track_analysis_edges_keys["speed"]
        self.displacement_key = self.track_analysis_edges_keys["displacement"]
        self.edge_time_key = self.track_analysis_edges_keys["edge_time"]
        self.edge_x_location_key = self.track_analysis_edges_keys["edge_x_location"]
        self.edge_y_location_key = self.track_analysis_edges_keys["edge_y_location"]
        self.edge_z_location_key = self.track_analysis_edges_keys["edge_z_location"]
        
        self.unique_tracks = {}
        self.unique_track_properties = {}
        self.unique_fft_properties = {}
        self.unique_spot_properties = {}
        self.channel_unique_spot_properties = {}
        self.edge_target_lookup = {}
        self.edge_source_lookup = {}
        self.generation_dict = {}
        self.tracklet_dict = {}
        self.graph_split = {}
        self.graph_tracks = {}


        self._get_xml_data()
        self._get_attributes()
        self._temporal_plots_trackmate()
        
              


    def _create_channel_tree(self):
          self._timed_channel_seg_image = {}
          if self.image is not None:
                intensity_image = self.image
          else:
                intensity_image = self.channel_seg_image

          for i in range(self.channel_seg_image.shape[0]):
                
                properties = regionprops(self.channel_seg_image[i,:], intensity_image[i,:])
                centroids = [prop.centroid for prop in properties]
                labels = [prop.label for prop in properties]
                volume = [prop.area for prop in properties]
                intensity_mean = [prop.intensity_mean for prop in properties]
                intensity_total = [prop.intensity_mean * prop.area for prop in properties]

                tree = spatial.cKDTree(centroids)

                self._timed_channel_seg_image[str(i)] =  tree, centroids, labels, volume, intensity_mean, intensity_total
          

    def _get_attributes(self):
            
             self.Attributeids, self.AllValues =  get_spot_dataset(self.spot_dataset, self.track_analysis_spot_keys, self.xcalibration, self.ycalibration, self.zcalibration, self.AttributeBoxname)
        
             self.TrackAttributeids, self.AllTrackValues = get_track_dataset(self.track_dataset,  self.track_analysis_spot_keys, self.track_analysis_track_keys, self.TrackAttributeBoxname)
             
             self.AllEdgesValues = get_edges_dataset(self.edges_dataset, self.edges_dataset_index, self.track_analysis_spot_keys, self.track_analysis_edges_keys)
        
    def _get_boundary_points(self):
         
        if  self.mask is not None and self.image is not None:
                    if len(self.mask.shape) < len(self.image.shape):
                        self.update_mask = np.zeros(
                            [
                                self.image.shape[0],
                                self.image.shape[1],
                                self.image.shape[2],
                                self.image.shape[3],
                            ]
                        )
                        for i in range(0, self.update_mask.shape[0]):
                            for j in range(0, self.update_mask.shape[1]):

                                self.update_mask[i, j, :, :] = self.mask[i, :, :]
                                
                    else:
                        self.update_mask = self.mask
                        
                    self.mask = self.update_mask.astype('uint16')

                    self.timed_mask, self.boundary = boundary_points(self.mask, self.xcalibration, self.ycalibration, self.zcalibration)
        else:
                    self.timed_mask = None
                    self.boundary = None

    #def _exchange_xml(self, frame):
          


    def _generate_generations(self, track):
         
        all_source_ids = []
        all_target_ids = [] 


        for edge in track.findall('Edge'):

                            source_id = edge.get(self.spot_source_id_key)
                            target_id = edge.get(self.spot_target_id_key)
                            all_source_ids.append(source_id)
                            all_target_ids.append(target_id)
                            
                            if source_id in self.edge_target_lookup.keys():
                               self.edge_target_lookup[source_id].append(target_id)
                            else:      
                               self.edge_target_lookup[source_id] = [target_id]

                            
                            self.edge_source_lookup[target_id] = source_id 

        return all_source_ids, all_target_ids 


    def _create_generations(self, all_source_ids, all_target_ids):
         
        root_leaf = []
        root_root = []
        root_splits = []
        split_count = 0
        #Get the root id
        for source_id in all_source_ids:
              if source_id not in all_target_ids:
                   root_root.append(source_id) 
                   self.tracklet_dict[source_id] = 0
                   
       
        #Get the leafs and splits     
        for target_id in all_target_ids:
             
             if target_id not in all_source_ids:
                  root_leaf.append(target_id)
                  self.tracklet_dict[target_id] = 0
             split_count = all_source_ids.count(target_id)
             if split_count > 1:
                      root_splits.append(target_id)
             self.tracklet_dict[target_id] = 0     

             
        #print('root and splits',root_root, root_leaf, root_splits)
        self._distance_root_leaf(root_root, root_leaf, root_splits)

        return root_root, root_splits, root_leaf


    def _iterate_split_down(self, root_leaf, root_splits):
         
         tracklet_before = 0
         for root_split in root_splits:
              
              target_cells = self.edge_target_lookup[root_split]
              for i in range(len(target_cells)):
                   
                   target_cell_id = target_cells[i]
                   self.graph_split[target_cell_id] = root_split 

                   target_cell_tracklet_id = i + 1 + tracklet_before
                   tracklet_before = tracklet_before + 1
                   self._assign_tracklet_id(target_cell_id, target_cell_tracklet_id, root_leaf, root_splits)

   
    def _assign_tracklet_id(self, target_cell_id, target_cell_tracklet_id, root_leaf, root_splits):
         
         if target_cell_id not in root_splits:
              self.tracklet_dict[target_cell_id] = target_cell_tracklet_id
              if target_cell_id not in root_leaf:
                 target_cell_id = self.edge_target_lookup[target_cell_id]
                 self._assign_tracklet_id(target_cell_id[0], target_cell_tracklet_id, root_leaf, root_splits)
                      
  
         


    def _distance_root_leaf(self, root_root, root_leaf, root_splits):


        
         #Generation 0
         root_cell_id = root_root[0]    
         self.generation_dict[root_cell_id] = '0'
         max_generation = len(root_splits)
         #Generation > 1
         for pre_leaf in root_leaf:
              if pre_leaf == root_cell_id:
                   self.generation_dict[pre_leaf] = '0'
              else:
                   self.generation_dict[pre_leaf] = str(max_generation) 
                   source_id = self.edge_source_lookup[pre_leaf]
                   if source_id in root_splits:
                         self.generation_dict[source_id] = str(max_generation - 1)
                   if source_id not in root_splits and source_id not in root_root:                         
                         source_id = self._recursive_path(source_id, root_splits, root_root, max_generation, gen_count = int(max_generation))

                   
                   
                              
    #Assign generation ID to each cell               
    def _recursive_path(self, source_id, root_splits, root_root, max_generation, gen_count ):
         

        if source_id not in root_root:  
            if source_id not in root_splits:
                            
                            self.generation_dict[source_id] = str(gen_count)
                            source_id = self.edge_source_lookup[source_id]
                            self._recursive_path(source_id, root_splits, root_root, max_generation, gen_count = gen_count)
            if source_id in root_splits:
                                    
                                    gen_count = gen_count - 1
                                    self.generation_dict[source_id] = str(gen_count)
                                    source_id = self.edge_source_lookup[source_id]
                                   
                                    self._recursive_path(source_id, root_splits, root_root, max_generation, gen_count = int(gen_count))

                                    
                                    
                            
    def _get_boundary_dist(self, frame, testlocation, cellradius):
         
        if self.mask is not None:

                tree, indices, masklabel, masklabelvolume = self.timed_mask[str(int(float(frame)))]
                if len(self.mask.shape) == 4:
                        z, y, x = testlocation 
                        region_label = self.mask[
                            int(float(frame) ),
                            int(float(z) / self.zcalibration),
                            int(float(y) / self.ycalibration),
                            int(float(x) / self.xcalibration),
                        ]
                if len(self.mask.shape) == 3:
                        y,x = testlocation
                        region_label = self.mask[
                            int(float(frame) ),
                            int(float(y) / self.ycalibration),
                            int(float(x) / self.xcalibration),
                        ]
                for k in range(0, len(masklabel)):
                    currentlabel = masklabel[k]
                    currentvolume = masklabelvolume[k]
                    currenttree = tree[k]
                    # Get the location and distance to the nearest boundary point
                    distance_cell_mask, location = currenttree.query(testlocation)
                    distance_cell_mask = max(0, distance_cell_mask - float(cellradius))
                   
        else:
                distance_cell_mask = 0

        return distance_cell_mask        
         
    

    def _get_xml_data(self):


                

                self.xml_content = et.fromstring(codecs.open(self.xml_path, "r", "utf8").read())
                
                if self.channel_seg_image is not None:
                      self.channel_xml_content = self.xml_content
                      self.xml_tree = et.parse(self.xml_path)
                      self.xml_root = self.xml_tree.getroot()
                      self.channel_xml_name = 'second_channel_' + os.path.splitext(os.path.basename(self.xml_path))[0] + '.xml'
                      self.channel_xml_path = os.path.dirname(self.xml_path)
                      self._create_channel_tree()
                       
                self.unique_objects = {}
                self.unique_properties = {}
                self.AllTrackIds = []
                self.DividingTrackIds = []
                self.NormalTrackIds = []
                self.all_track_properties = []
                self.split_points_times = []

                self.filtered_track_ids = [
                    int(track.get(self.trackid_key))
                    for track in self.xml_content.find("Model")
                    .find("FilteredTracks")
                    .findall("TrackID")
                ]
                
                self.AllTrackIds.append(None)
                self.DividingTrackIds.append(None)
                self.NormalTrackIds.append(None)
                
                self.AllTrackIds.append(self.TrackidBox)
                self.DividingTrackIds.append(self.TrackidBox)
                self.NormalTrackIds.append(self.TrackidBox)
                
                
                self.Spotobjects = self.xml_content.find('Model').find('AllSpots')
                # Extract the tracks from xml
                self.tracks = self.xml_content.find("Model").find("AllTracks")
                self.settings = self.xml_content.find("Settings").find("ImageData")
                self.xcalibration = float(self.settings.get("pixelwidth"))
                self.ycalibration = float(self.settings.get("pixelheight"))
                self.zcalibration = float(self.settings.get("voxeldepth"))
                self.tcalibration = int(float(self.settings.get("timeinterval")))
                self._get_boundary_points()

                for frame in self.Spotobjects.findall('SpotsInFrame'):

                    for Spotobject in frame.findall('Spot'):
                        # Create object with unique cell ID
                        cell_id = int(Spotobject.get(self.spotid_key))
                        # Get the TZYX location of the cells in that frame
                        TOTAL_INTENSITY_CH1 = Spotobject.get(self.total_intensity_ch1_key)
                        MEAN_INTENSITY_CH1 = Spotobject.get(self.mean_intensity_ch1_key)
                        TOTAL_INTENSITY_CH2 = Spotobject.get(self.total_intensity_ch2_key)
                        MEAN_INTENSITY_CH2 = Spotobject.get(self.mean_intensity_ch2_key)
                        RADIUS = Spotobject.get(self.radius_key)
                        QUALITY = Spotobject.get(self.quality_key)
                        testlocation = (Spotobject.get(self.zposid_key), Spotobject.get(self.yposid_key),  Spotobject.get(self.xposid_key))
                        frame = Spotobject.get(self.frameid_key)
                        distance_cell_mask = self._get_boundary_dist(frame, testlocation, RADIUS)
                        
                        self.unique_spot_properties[cell_id] = {
                            self.cellid_key: int(cell_id), 
                            self.frameid_key : int(float(Spotobject.get(self.frameid_key))),
                            self.zposid_key : round(float(Spotobject.get(self.zposid_key)), 3),
                            self.yposid_key : round(float(Spotobject.get(self.yposid_key)), 3),
                            self.xposid_key : round(float(Spotobject.get(self.xposid_key)), 3),
                            self.total_intensity_ch1_key : round(float(TOTAL_INTENSITY_CH1)),
                            self.mean_intensity_ch1_key : round(float(MEAN_INTENSITY_CH1)),
                            self.total_intensity_ch2_key : round(float(TOTAL_INTENSITY_CH2)),
                            self.mean_intensity_ch2_key : round(float(MEAN_INTENSITY_CH2)),
                            self.radius_key : round(float(RADIUS)),
                            self.quality_key : round(float(QUALITY)),
                            self.distance_cell_mask_key: round(float(distance_cell_mask),2)
                        }
            
                        if self.channel_seg_image is not None:
                                    tree, centroids, labels, volume, intensity_mean, intensity_total = self._timed_channel_seg_image[str(int(float(frame)))]
                                    dist, index = tree.query(testlocation)
                                    location = (int(centroids[index][0]), int(centroids[index][1]), int(centroids[index][2]))
                                    QUALITY = volume[index]
                                    RADIUS = math.pow(QUALITY, 1.0/3.0) * self.xcalibration * self.ycalibration * self.zcalibration
                                    distance_cell_mask = self._get_boundary_dist(frame, location, RADIUS)
                                    self.channel_unique_spot_properties[cell_id] = {
                                            self.cellid_key: int(cell_id), 
                                            self.frameid_key : int(float(Spotobject.get(self.frameid_key))),
                                            self.zposid_key : round(float(centroids[index][0]), 3),
                                            self.yposid_key : round(float(centroids[index][1]), 3),
                                            self.xposid_key : round(float(centroids[index][2]), 3),

                                            self.total_intensity_ch1_key : round(float(intensity_total[index])),
                                            self.mean_intensity_ch1_key : round(float(intensity_mean[index])),
                                            self.total_intensity_ch2_key : round(float(intensity_total[index])),
                                            self.mean_intensity_ch2_key : round(float(intensity_mean[index])),

                                            self.radius_key : round(float(RADIUS)),
                                            self.quality_key : round(float(QUALITY)),
                                            self.distance_cell_mask_key: round(float(distance_cell_mask),2)

                                    } 

                if self.channel_seg_image is not None:                    
                        for Spotobject in self.xml_root.iter('Spot'):
                                cell_id = int(Spotobject.get(self.spotid_key))
                                
 
                                new_positionx =  self.channel_unique_spot_properties[cell_id][self.xposid_key]
                                new_positiony =  self.channel_unique_spot_properties[cell_id][self.yposid_key]
                                new_positionz =  self.channel_unique_spot_properties[cell_id][self.zposid_key]

                                new_total_intensity_ch1 = self.channel_unique_spot_properties[cell_id][self.total_intensity_ch1_key]
                                new_mean_intensity_ch1 = self.channel_unique_spot_properties[cell_id][self.mean_intensity_ch1_key]
                                new_total_intensity_ch2 = self.channel_unique_spot_properties[cell_id][self.total_intensity_ch2_key]
                                new_mean_intensity_ch2 = self.channel_unique_spot_properties[cell_id][self.mean_intensity_ch2_key]

                                new_radius = self.channel_unique_spot_properties[cell_id][self.radius_key]
                                new_quality = self.channel_unique_spot_properties[cell_id][self.quality_key]
                                new_distance_cell_mask = self.channel_unique_spot_properties[cell_id][self.distance_cell_mask_key]

                                Spotobject.set(self.xposid_key, str(new_positionx))     
                                Spotobject.set(self.yposid_key, str(new_positiony))
                                Spotobject.set(self.zposid_key, str(new_positionz))

                                Spotobject.set(self.total_intensity_ch1_key, str(new_total_intensity_ch1))     
                                Spotobject.set(self.mean_intensity_ch1_key, str(new_mean_intensity_ch1))
                                Spotobject.set(self.total_intensity_ch2_key, str(new_total_intensity_ch2))
                                Spotobject.set(self.mean_intensity_ch2_key, str(new_mean_intensity_ch2))

                                Spotobject.set(self.radius_key, str(new_radius))     
                                Spotobject.set(self.quality_key, str(new_quality))
                                Spotobject.set(self.distance_cell_mask_key, str(new_distance_cell_mask))

                self.xml_tree.write(os.path.join(self.channel_xml_path, self.channel_xml_name))



                for track in self.tracks.findall('Track'):

                    track_id = int(track.get(self.trackid_key))

                    if track_id in self.filtered_track_ids:
                        
                            current_cell_ids = []
                            unique_tracklet_ids = []
                            current_tracklets = {}
                            current_tracklets_properties = {}
                            all_source_ids, all_target_ids =  self._generate_generations(track)
                            root_root, root_splits, root_leaf = self._create_generations(all_source_ids, all_target_ids) 

                            self._iterate_split_down(root_leaf, root_splits)
                            for edge in track.findall('Edge'):
                                  

                                source_id = edge.get(self.spot_source_id_key)
                                target_id = edge.get(self.spot_target_id_key)

                                
                                #Root 
                                if int(source_id) not in all_target_ids:
                                        self._dict_update(unique_tracklet_ids, current_cell_ids, edge, source_id, track_id, None, target_id)

                                #Leaf
                                if int(target_id) not in all_source_ids:
                                        self._dict_update(unique_tracklet_ids, current_cell_ids, edge, target_id, track_id, source_id, None)

                                       
                                #All other types
                                else:
                                        self._dict_update(unique_tracklet_ids, current_cell_ids, edge, source_id, track_id, source_id, target_id)
                                        
                                      
                                                 

                                # Determine if a track has divisions or none
                                if len(root_splits) > 0:
                                    DividingTrajectory = True
                                    if str(track_id) not in self.AllTrackIds:
                                       self.AllTrackIds.append(str(track_id))
                                    if str(track_id) not in self.DividingTrackIds:     
                                      self.DividingTrackIds.append(str(track_id))
                                         
                                else:
                                    DividingTrajectory = False
                                    if str(track_id) not in self.AllTrackIds:
                                        self.AllTrackIds.append(str(track_id))
                                    if str(track_id) not in self.NormalTrackIds:    
                                       self.NormalTrackIds.append(str(track_id))

                                if int(source_id) not in all_target_ids:
                                        self.unique_spot_properties[int(source_id)].update({self.dividing_key : DividingTrajectory})
                                if int(target_id) not in all_source_ids:
                                        self.unique_spot_properties[int(target_id)].update({self.dividing_key : DividingTrajectory})    
                            
                           

                            for (k,v) in self.unique_spot_properties.items():
                                if int(k) in current_cell_ids:
                                  all_dict_values = self.unique_spot_properties[k]
                                  unique_id = str(all_dict_values[self.uniqueid_key])
                                  current_track_id = str(all_dict_values[self.trackid_key])
                                  t = int(float(all_dict_values[self.frameid_key]))
                                  z = float(all_dict_values[self.zposid_key])
                                  y = float(all_dict_values[self.yposid_key])
                                  x = float(all_dict_values[self.xposid_key])
                                  gen_id = int(float(all_dict_values[self.generationid_key]))
                                  speed = float(all_dict_values[self.speed_key])
                                  dcr = float(all_dict_values[self.directional_change_rate_key])
                                  dcr = scale_value(float(dcr))
                                  speed = scale_value(float(speed))
                                  total_intensity_ch1 =  float(all_dict_values[self.total_intensity_ch1_key])
                                  total_intensity_ch2 =  float(all_dict_values[self.total_intensity_ch2_key])
                                  volume_pixels = int(float(all_dict_values[self.quality_key]))
                                  if current_track_id in current_tracklets:
                                     tracklet_array = current_tracklets[current_track_id]
                                     current_tracklet_array = np.array([int(float(unique_id)), t, z/self.zcalibration, y/self.ycalibration, x/self.xcalibration])
                                     current_tracklets[current_track_id] = np.vstack((tracklet_array, current_tracklet_array))

                                     value_array = current_tracklets_properties[current_track_id]
                                     current_value_array = np.array([t, gen_id, speed, dcr, total_intensity_ch1, total_intensity_ch2, volume_pixels])
                                     current_tracklets_properties[current_track_id] = np.vstack((value_array, current_value_array))

                                  else:
                                     current_tracklet_array = np.array([int(float(unique_id)), t, z/self.zcalibration, y/self.ycalibration, x/self.xcalibration])
                                     current_tracklets[current_track_id] = current_tracklet_array 

                                     current_value_array = np.array([t, gen_id, speed, dcr, total_intensity_ch1, total_intensity_ch2, volume_pixels])
                                     current_tracklets_properties[current_track_id] = current_value_array
                                           

                           
                          
                            current_tracklets = np.asarray(current_tracklets[str(track_id)])
                            current_tracklets_properties = np.asarray(current_tracklets_properties[str(track_id)])
                            
                            self.unique_tracks[track_id] = current_tracklets     
                            self.unique_track_properties[track_id] = current_tracklets_properties
         
                self._compute_fourier()
                for (k,v) in self.graph_split.items():
                           
                            daughter_track_id =  int(float(str(self.unique_spot_properties[int(float(k))][self.uniqueid_key])))
                            parent_track_id = int(float(str(self.unique_spot_properties[int(float(v))][self.uniqueid_key])))
                            self.graph_tracks[daughter_track_id] = parent_track_id
                            
                
    def _compute_fourier(self):

          for (k,v) in self.unique_tracks.items():
                
                track_id = k
                tracklets = v 
                tracklet_properties = self.unique_track_properties[k] 
                intensity_ch1 = tracklet_properties[:,-4:-3]
                intensity_ch2 = tracklet_properties[:,-3:-2]
                time = tracklet_properties[:,0:1]

                point_sample_ch1 = intensity_ch1.shape[0]
                if point_sample_ch1 > 0:
                            xf_sample_ch1 = fftfreq(point_sample_ch1, self.tcalibration)
                            fftstrip_sample_ch1 = fft(intensity_ch1)
                            ffttotal_sample_ch1 = np.abs(fftstrip_sample_ch1)
                            xf_sample_ch1 = xf_sample_ch1[0 : len(xf_sample_ch1) // 2]
                            ffttotal_sample_ch1 = ffttotal_sample_ch1[0 : len(ffttotal_sample_ch1) // 2]

                point_sample_ch2 = intensity_ch2.shape[0]
                if point_sample_ch2 > 0:
                            xf_sample_ch2 = fftfreq(point_sample_ch2, self.tcalibration)
                            fftstrip_sample_ch2 = fft(intensity_ch2)
                            ffttotal_sample_ch2 = np.abs(fftstrip_sample_ch2)
                            xf_sample_ch2 = xf_sample_ch2[0 : len(xf_sample_ch2) // 2]
                            ffttotal_sample_ch2 = ffttotal_sample_ch2[0 : len(ffttotal_sample_ch2) // 2] 

                self.unique_fft_properties[track_id] = time, xf_sample_ch1, ffttotal_sample_ch1, xf_sample_ch2, ffttotal_sample_ch2

                                 
    def _dict_update(self, unique_tracklet_ids: List, current_cell_ids: List, edge, cell_id, track_id, source_id, target_id):

        generation_id = self.generation_dict[cell_id]
        tracklet_id = self.tracklet_dict[cell_id]

        unique_id = str(track_id) + str(generation_id) + str(tracklet_id)
        unique_tracklet_ids.append(str(unique_id))
    
        current_cell_ids.append(int(cell_id))
        self.unique_spot_properties[int(cell_id)].update({self.uniqueid_key : str(unique_id)})
        self.unique_spot_properties[int(cell_id)].update({self.trackletid_key : str(tracklet_id)}) 
        self.unique_spot_properties[int(cell_id)].update({self.generationid_key : str(generation_id)}) 
        self.unique_spot_properties[int(cell_id)].update({self.trackid_key : str(track_id)})
        self.unique_spot_properties[int(cell_id)].update({self.directional_change_rate_key : 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.speed_key : 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.acceleration_key : 0.0})

        if source_id is not None:
            self.unique_spot_properties[int(cell_id)].update({self.beforeid_key : int(source_id)})
            vec_1 = [float(self.unique_spot_properties[int(cell_id)][self.xposid_key]) - float(self.unique_spot_properties[int(source_id)][self.xposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.yposid_key]) - float(self.unique_spot_properties[int(source_id)][self.yposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.zposid_key]) -  float(self.unique_spot_properties[int(source_id)][self.zposid_key])]
            speed = np.sqrt(np.dot(vec_1, vec_1))/self.tcalibration
            self.unique_spot_properties[int(cell_id)].update({self.speed_key : round(speed, 3)})
            if str(source_id) in self.edge_source_lookup:
                    pre_source_id = self.edge_source_lookup[str(source_id)]
                    
                    vec_0 = [float(self.unique_spot_properties[int(source_id)][self.xposid_key]) - float(self.unique_spot_properties[int(pre_source_id)][self.xposid_key]), 
                            float(self.unique_spot_properties[int(source_id)][self.yposid_key]) - float(self.unique_spot_properties[int(pre_source_id)][self.yposid_key]), 
                            float(self.unique_spot_properties[int(source_id)][self.zposid_key]) -  float(self.unique_spot_properties[int(pre_source_id)][self.zposid_key])]
                    
                    vec_2 = [float(self.unique_spot_properties[int(cell_id)][self.xposid_key]) - 2 * float(self.unique_spot_properties[int(source_id)][self.xposid_key]) + float(self.unique_spot_properties[int(pre_source_id)][self.xposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.yposid_key]) - 2 * float(self.unique_spot_properties[int(source_id)][self.yposid_key]) + float(self.unique_spot_properties[int(pre_source_id)][self.yposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.zposid_key]) -  2 * float(self.unique_spot_properties[int(source_id)][self.zposid_key]) + float(self.unique_spot_properties[int(pre_source_id)][self.zposid_key])]
                    acc = np.sqrt(np.dot(vec_2, vec_2))/self.tcalibration
                    angle = angular_change(vec_0, vec_1)
                    self.unique_spot_properties[int(cell_id)].update({self.directional_change_rate_key : round(angle, 3)})
                    self.unique_spot_properties[int(cell_id)].update({self.acceleration_key : round(acc, 3)})
        else:
            self.unique_spot_properties[int(cell_id)].update({self.beforeid_key : None}) 

        if target_id is not None:       
            self.unique_spot_properties[int(cell_id)].update({self.afterid_key : int(target_id)}) 
        else:
            self.unique_spot_properties[int(cell_id)].update({self.afterid_key : None})
                 
                                    
                
    def _temporal_plots_trackmate(self):
    
    
    
                self.Attr = {}
                sourceid_key = self.track_analysis_edges_keys["spot_source_id"]
                dcr_key = self.track_analysis_edges_keys["directional_change_rate"]
                speed_key = self.track_analysis_edges_keys["speed"]
                disp_key = self.track_analysis_edges_keys["displacement"]
                starttime = int(min(self.AllValues[self.frameid_key]))
                endtime = int(max(self.AllValues[self.frameid_key]))
                 
                self.time = []
                self.mitotic_mean_disp_z = []
                self.mitotic_var_disp_z = []

                self.mitotic_mean_disp_y = []
                self.mitotic_var_disp_y = []

                self.mitotic_mean_disp_x = []
                self.mitotic_var_disp_x = []

                self.mitotic_mean_radius = []
                self.mitotic_var_radius = []

                self.mitotic_mean_speed = []
                self.mitotic_var_speed = []

                self.mitotic_mean_directional_change = []
                self.mitotic_var_directional_change = []

                self.non_mitotic_mean_disp_z = []
                self.non_mitotic_var_disp_z = []

                self.non_mitotic_mean_disp_y = []
                self.non_mitotic_var_disp_y = []

                self.non_mitotic_mean_disp_x = []
                self.non_mitotic_var_disp_x = []

                self.non_mitotic_mean_radius = []
                self.non_mitotic_var_radius = []

                self.non_mitotic_mean_speed = []
                self.non_mitotic_var_speed = []

                self.non_mitotic_mean_directional_change = []
                self.non_mitotic_var_directional_change = []

                self.all_mean_disp_z = []
                self.all_var_disp_z = []

                self.all_mean_disp_y = []
                self.all_var_disp_y = []

                self.all_mean_disp_x = []
                self.all_var_disp_x = []

                self.all_mean_radius = []
                self.all_var_radius = []

                self.all_mean_speed = []
                self.all_var_speed = []

                self.all_mean_directional_change = []
                self.all_var_directional_change = []




                all_spots_tracks = {}
                for (k,v) in self.unique_spot_properties.items():
                      
                      all_spots = self.unique_spot_properties[k]
                      if self.trackid_key in all_spots:
                          all_spots_tracks[k] = all_spots
                
                for i in tqdm(range(starttime, endtime), total=endtime - starttime):
                      
                    mitotic_disp_z = []
                    mitotic_disp_y = []
                    mitotic_disp_x = []
                    mitotic_radius = []
                    mitotic_speed = []
                    mitotic_directional_change = []

                    non_mitotic_disp_z = []
                    non_mitotic_disp_y = []
                    non_mitotic_disp_x = []
                    non_mitotic_radius = []
                    non_mitotic_speed = []
                    non_mitotic_directional_change = []
 
                    all_disp_z = []
                    all_disp_y = []
                    all_disp_x = []
                    all_radius = []
                    all_speed = []
                    all_directional_change = []


                    for (k,v) in all_spots_tracks.items():
                            
                            current_time = all_spots_tracks[k][self.frameid_key]
                            mitotic = all_spots_tracks[k][self.dividing_key]
                            if i == int(current_time):
                                  if mitotic:
                                        mitotic_disp_z.append(all_spots_tracks[k][self.zposid_key])
                                        mitotic_disp_y.append(all_spots_tracks[k][self.yposid_key])
                                        mitotic_disp_x.append(all_spots_tracks[k][self.xposid_key])
                                        mitotic_radius.append(all_spots_tracks[k][self.radius_key])
                                        mitotic_speed.append(all_spots_tracks[k][self.speed_key])
                                        mitotic_directional_change.append(all_spots_tracks[k][self.directional_change_rate_key])
                                  if not mitotic:
                                        non_mitotic_disp_z.append(all_spots_tracks[k][self.zposid_key])
                                        non_mitotic_disp_y.append(all_spots_tracks[k][self.yposid_key])
                                        non_mitotic_disp_x.append(all_spots_tracks[k][self.xposid_key])
                                        non_mitotic_radius.append(all_spots_tracks[k][self.radius_key])
                                        non_mitotic_speed.append(all_spots_tracks[k][self.speed_key])
                                        non_mitotic_directional_change.append(all_spots_tracks[k][self.directional_change_rate_key])

                                  all_disp_z.append(all_spots_tracks[k][self.zposid_key])
                                  all_disp_y.append(all_spots_tracks[k][self.yposid_key])
                                  all_disp_x.append(all_spots_tracks[k][self.xposid_key])
                                  all_radius.append(all_spots_tracks[k][self.radius_key])
                                  all_speed.append(all_spots_tracks[k][self.speed_key])
                                  all_directional_change.append(all_spots_tracks[k][self.directional_change_rate_key])       
                                              

                    mitotic_disp_z = np.abs(np.diff(mitotic_disp_z))
                    mitotic_disp_y = np.abs(np.diff(mitotic_disp_y))
                    mitotic_disp_x = np.abs(np.diff(mitotic_disp_x))

                    non_mitotic_disp_z = np.abs(np.diff(non_mitotic_disp_z))
                    non_mitotic_disp_y = np.abs(np.diff(non_mitotic_disp_y))
                    non_mitotic_disp_x = np.abs(np.diff(non_mitotic_disp_x))

                    all_disp_z = np.abs(np.diff(all_disp_z))
                    all_disp_y = np.abs(np.diff(all_disp_y))
                    all_disp_x = np.abs(np.diff(all_disp_x))


                                        
                    self.time.append(i * self.tcalibration)

                    self.mitotic_mean_disp_z.append(np.mean(mitotic_disp_z))
                    self.mitotic_var_disp_z.append(np.std(mitotic_disp_z))

                    self.mitotic_mean_disp_y.append(np.mean(mitotic_disp_y))
                    self.mitotic_var_disp_y.append(np.std(mitotic_disp_y))

                    self.mitotic_mean_disp_x.append(np.mean(mitotic_disp_x))
                    self.mitotic_var_disp_x.append(np.std(mitotic_disp_x))

                    self.mitotic_mean_radius.append(np.mean(mitotic_radius))
                    self.mitotic_var_radius.append(np.std(mitotic_radius))

                    self.mitotic_mean_speed.append(np.mean(mitotic_speed))
                    self.mitotic_var_speed.append(np.std(mitotic_speed))

                    self.mitotic_mean_directional_change.append(np.mean(mitotic_directional_change))
                    self.mitotic_var_directional_change.append(np.std(mitotic_directional_change))

                    self.non_mitotic_mean_disp_z.append(np.mean(non_mitotic_disp_z))
                    self.non_mitotic_var_disp_z.append(np.std(non_mitotic_disp_z))

                    self.non_mitotic_mean_disp_y.append(np.mean(non_mitotic_disp_y))
                    self.non_mitotic_var_disp_y.append(np.std(non_mitotic_disp_y))

                    self.non_mitotic_mean_disp_x.append(np.mean(non_mitotic_disp_x))
                    self.non_mitotic_var_disp_x.append(np.std(non_mitotic_disp_x))

                    self.non_mitotic_mean_radius.append(np.mean(non_mitotic_radius))
                    self.non_mitotic_var_radius.append(np.std(non_mitotic_radius))

                    self.non_mitotic_mean_speed.append(np.mean(non_mitotic_speed))
                    self.non_mitotic_var_speed.append(np.std(non_mitotic_speed))

                    self.non_mitotic_mean_directional_change.append(np.mean(non_mitotic_directional_change))
                    self.non_mitotic_var_directional_change.append(np.std(non_mitotic_directional_change)) 

                    self.all_mean_disp_z.append(np.mean(all_disp_z))
                    self.all_var_disp_z.append(np.std(all_disp_z))

                    self.all_mean_disp_y.append(np.mean(all_disp_y))
                    self.all_var_disp_y.append(np.std(all_disp_y))

                    self.all_mean_disp_x.append(np.mean(all_disp_x))
                    self.all_var_disp_x.append(np.std(all_disp_x))

                    self.all_mean_radius.append(np.mean(all_radius))
                    self.all_var_radius.append(np.std(all_radius))

                    self.all_mean_speed.append(np.mean(all_speed))
                    self.all_var_speed.append(np.std(all_speed))

                    self.all_mean_directional_change.append(np.mean(all_directional_change))
                    self.all_var_directional_change.append(np.std(all_directional_change))
                            
        
def boundary_points(mask, xcalibration, ycalibration, zcalibration):

    ndim = len(mask.shape)
    timed_mask = {}
    # YX shaped object
    if ndim == 2:
        mask = label(mask)
        labels = []
        size = []
        tree = []
        properties = regionprops(mask, mask)
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
            properties = regionprops(mask[i, :], mask[i, :])
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
        
        results.append(parallel_map(timed_mask,mask, xcalibration, ycalibration, zcalibration, Boundary, i) for i in tqdm(x_ls))
        da.delayed(results).compute()
        

    return timed_mask, Boundary        

def parallel_map(timed_mask, mask, xcalibration, ycalibration, zcalibration, Boundary, i):

    mask[i, :] = label(mask[i, :])
    properties = regionprops(mask[i, :], mask[i, :])
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

 


def sortTracks(List):

    return int(float(List[2]))


def get_csv_data(csv):

        dataset = pd.read_csv(
            csv, delimiter=",", encoding="unicode_escape", low_memory=False
        )[3:]
        dataset_index = dataset.index
        return dataset, dataset_index
    
def get_spot_dataset(spot_dataset, track_analysis_spot_keys, xcalibration, ycalibration, zcalibration, AttributeBoxname):
        AllValues = {}
        posix = track_analysis_spot_keys["posix"]
        posiy = track_analysis_spot_keys["posiy"]
        posiz = track_analysis_spot_keys["posiz"]
        frame = track_analysis_spot_keys["frame"]
        
        LocationX = (
            spot_dataset[posix].astype("float") / xcalibration
        ).astype("int")
        LocationY = (
            spot_dataset[posiy].astype("float") / ycalibration
        ).astype("int")
        LocationZ = (
            spot_dataset[posiz].astype("float") / zcalibration
        ).astype("int")
        LocationT = (spot_dataset[frame].astype("float")).astype("int")
        

        for (k,v) in track_analysis_spot_keys.items():
            

                AllValues[v] = spot_dataset[v].astype("float")

        AllValues[posix] = round(LocationX,3)
        AllValues[posiy] = round(LocationY,3)
        AllValues[posiz] = round(LocationZ,3)
        AllValues[frame] = round(LocationT,3)
        Attributeids = []
        Attributeids.append(AttributeBoxname)
        for attributename in track_analysis_spot_keys.values():
              Attributeids.append(attributename)    
            
        
        return Attributeids, AllValues     
    
def get_track_dataset(track_dataset, track_analysis_spot_keys, track_analysis_track_keys, TrackAttributeBoxname):

        AllTrackValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        Tid = track_dataset[track_id].astype("float")
       
        AllTrackValues[track_id] = Tid
      
        for (k, v) in track_analysis_track_keys.items():

                x = track_dataset[v].astype("float")
                minval = min(x)
                maxval = max(x)

                if minval > 0 and maxval <= 1:

                    x = x + 1

                AllTrackValues[k] = round(x, 3)

        TrackAttributeids = []
        TrackAttributeids.append(TrackAttributeBoxname)
        for attributename in track_analysis_track_keys.keys():
            TrackAttributeids.append(attributename)    
    
        return TrackAttributeids, AllTrackValues
    
def get_edges_dataset(edges_dataset, edges_dataset_index, track_analysis_spot_keys, track_analysis_edges_keys):

        AllEdgesValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        Tid = edges_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = edges_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1
        AllEdgesValues[track_id] = Tid

        for k in track_analysis_edges_keys.values():

            if k != track_id:
                x = edges_dataset[k].astype("float")

                AllEdgesValues[k] = x   
         
        return AllEdgesValues   
    
       
    
def scale_value(x, scale = 255 * 255):


     return x * scale   
    
def sortFirst(List):

    return int(float(List[0]))    

def prob_sigmoid(x):
    return 1 - math.exp(-x)


def angular_change(vec_0, vec_1):
        
        vec_0 = vec_0 / np.linalg.norm(vec_0)
        vec_1 = vec_1 / np.linalg.norm(vec_1)
        angle = np.arccos(np.clip(np.dot(vec_0, vec_1), -1.0, 1.0))
        angle = angle * 180 / np.pi
        return angle
     

           
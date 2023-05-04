
from tqdm import tqdm
import numpy as np 
import codecs
import lxml.etree as et
#import xml.etree.ElementTree as et
import pandas as pd
import math
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy import spatial
from typing import List
from scipy.fftpack import fft, fftfreq
import os
from pathlib import Path 
import concurrent
from .clustering import Clustering
from scipy import ndimage



class TrackMate(object):
    
    def __init__(self, xml_path, spot_csv_path, track_csv_path, edges_csv_path, AttributeBoxname, TrackAttributeBoxname, TrackidBox, axes,  progress_bar = None, 
                 master_xml_path: Path = None, master_extra_name = '', seg_image = None, channel_seg_image = None, image = None, mask = None, fourier = True, cluster_model = None, num_points = 2048, batch_size = 1):
        
        
        self.xml_path = xml_path
        self.master_xml_path = master_xml_path
        self.spot_csv_path = spot_csv_path
        self.track_csv_path = track_csv_path 
        self.edges_csv_path = edges_csv_path
        self.image = image 
        self.mask = mask
        self.fourier = fourier
        self.cluster_model = cluster_model 
        self.channel_seg_image = channel_seg_image
        self.seg_image = seg_image
        self.AttributeBoxname = AttributeBoxname
        self.TrackAttributeBoxname = TrackAttributeBoxname
        self.TrackidBox = TrackidBox
        self.master_extra_name = master_extra_name
       
       
        self.num_points = num_points
        self.spot_dataset, self.spot_dataset_index = get_csv_data(self.spot_csv_path)
        self.track_dataset, self.track_dataset_index = get_csv_data(self.track_csv_path)
        self.edges_dataset, self.edges_dataset_index = get_csv_data(self.edges_csv_path)
        self.progress_bar = progress_bar
        self.axes = axes                
        self.batch_size = batch_size 
        
                                             
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
                mean_intensity="MEAN_INTENSITY",
                total_intensity="TOTAL_INTENSITY"
            )
        self.track_analysis_edges_keys = dict(
                
                spot_source_id="SPOT_SOURCE_ID",
                spot_target_id="SPOT_TARGET_ID",
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
                linearity_forward_progression="LINEARITY_OF_FORWARD_PROGRESSION"
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
        self.number_dividing_key = 'number_dividing'
        self.distance_cell_mask_key = 'distance_cell_mask'
        self.maskcentroid_x_key = 'maskcentroid_x_key'
        self.maskcentroid_z_key = 'maskcentroid_z_key'
        self.maskcentroid_y_key = 'maskcentroid_y_key'
        self.cellaxis_mask_key = 'cellaxis_mask_key'
        self.cellid_key = 'cell_id'
        self.acceleration_key = 'acceleration'
        self.centroid_key = 'centroid'
        self.clusterclass_key = 'cluster_class'
        self.clusterscore_key = 'cluster_score'
        self.eccentricity_comp_firstkey = 'cloud_eccentricity_comp_first'
        self.eccentricity_comp_secondkey = 'cloud_eccentricity_comp_second'
        self.surface_area_key = 'cloud_surfacearea'
        self.radial_angle_key = 'radial_angle_key'
        self.motion_angle_key = 'motion_angle' 

        self.mean_intensity_ch1_key = self.track_analysis_spot_keys["mean_intensity_ch1"]
        self.mean_intensity_ch2_key = self.track_analysis_spot_keys["mean_intensity_ch2"]
        self.total_intensity_ch1_key = self.track_analysis_spot_keys["total_intensity_ch1"]
        self.total_intensity_ch2_key = self.track_analysis_spot_keys["total_intensity_ch2"]

        self.mean_intensity_key = self.track_analysis_spot_keys["mean_intensity"]
        self.total_intensity_key = self.track_analysis_spot_keys["total_intensity"]

        self.spot_source_id_key = self.track_analysis_edges_keys["spot_source_id"]
        self.spot_target_id_key = self.track_analysis_edges_keys["spot_target_id"]
       
        
        self.speed_key = self.track_analysis_edges_keys["speed"]
        self.displacement_key = self.track_analysis_edges_keys["displacement"]
        self.edge_time_key = self.track_analysis_edges_keys["edge_time"]
        self.edge_x_location_key = self.track_analysis_edges_keys["edge_x_location"]
        self.edge_y_location_key = self.track_analysis_edges_keys["edge_y_location"]
        self.edge_z_location_key = self.track_analysis_edges_keys["edge_z_location"]
        
        self.unique_tracks = {}
        self.unique_track_mitosis_label = {}
        self.unique_track_properties = {}
        self.unique_fft_properties = {}
        self.unique_cluster_properties = {}
        self.unique_shape_properties = {}
        self.unique_dynamic_properties = {}
        self.unique_spot_properties = {}
        self.unique_spot_centroid = {}
        self.unique_track_centroid = {}
        self.root_spots = {}
        self.all_current_cell_ids = {}
        self.channel_unique_spot_properties = {}
        self.edge_target_lookup = {}
        self.edge_source_lookup = {}
        self.generation_dict = {}
        self.tracklet_dict = {}
        self.graph_split = {}
        self.graph_tracks = {}
        self._timed_centroid = {}
        self.count = 0
        xml_parser = et.XMLParser(huge_tree=True)
        if self.master_xml_path is None:
               self.master_xml_path = Path('.')
        
        if self.master_xml_path.is_dir() and self.xml_path is not None:
                print('Reading XML')
                self.xml_content = et.fromstring(open(self.xml_path).read().encode(), xml_parser)
                
                self.filtered_track_ids = [
                            int(track.get(self.trackid_key))
                            for track in self.xml_content.find("Model")
                            .find("FilteredTracks")
                            .findall("TrackID")
                        ]
                self.max_track_id = max(self.filtered_track_ids)        
                
                self._get_xml_data()
        if not isinstance(self.master_xml_path, str):      
          if self.master_xml_path.is_file():
               print('Reading Master XML')
               
               self.xml_content = et.fromstring(open(self.master_xml_path).read().encode(), xml_parser)
               
               self.filtered_track_ids = [
                            int(track.get(self.trackid_key))
                            for track in self.xml_content.find("Model")
                            .find("FilteredTracks")
                            .findall("TrackID")
                        ]
               self.max_track_id = max(self.filtered_track_ids)        
                
               self._get_master_xml_data()
                       

     


    def _create_channel_tree(self):
          self._timed_channel_seg_image = {}
          self.count = 0
          futures = []
          with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                    for i in range(self.channel_seg_image.shape[0]):
                        futures.append(executor.submit(self._channel_computer, i))
          
                    if self.progress_bar is not None:
                                 
                                    self.progress_bar.label = "Doing channel computation"
                                    self.progress_bar.range = (
                                        0,
                                        len(futures),
                                    )
                                    self.progress_bar.show()

                    for r in concurrent.futures.as_completed(futures):
                                    self.count = self.count + 1
                                    if self.progress_bar is not None:
                                      self.progress_bar.value =  self.count
                                    r.result()

 

    def _channel_computer(self, i):
                
                if self.image is not None:
                        intensity_image = self.image
                else:
                        intensity_image = self.channel_seg_image
                
                properties = regionprops(self.channel_seg_image[i,:], intensity_image[i,:])
                centroids = [prop.centroid for prop in properties]
                labels = [prop.label for prop in properties]
                volume = [prop.area for prop in properties]
                intensity_mean = [prop.intensity_mean for prop in properties]
                intensity_total = [prop.intensity_mean * prop.area for prop in properties]
                bounding_boxes = [prop.bbox for prop in properties]

                tree = spatial.cKDTree(centroids)

                self._timed_channel_seg_image[str(i)] =  tree, centroids, labels, volume, intensity_mean, intensity_total, bounding_boxes
          

    def _get_attributes(self):
            
             self.Attributeids, self.AllValues =  get_spot_dataset(self.spot_dataset, self.track_analysis_spot_keys, self.xcalibration, self.ycalibration, self.zcalibration, self.AttributeBoxname, self.detectorchannel)
             print('obtianed spot attributes')
             self.TrackAttributeids, self.AllTrackValues = get_track_dataset(self.track_dataset,  self.track_analysis_spot_keys, self.track_analysis_track_keys, self.TrackAttributeBoxname)
             print('obtained track attributes')
             self.AllEdgesValues = get_edges_dataset(self.edges_dataset, self.edges_dataset_index, self.track_analysis_spot_keys, self.track_analysis_edges_keys)
             print('obtained edge attributes')

    
                    
    def _get_boundary_points(self):
         
        print('Computing boundary points') 
        if  self.mask is not None:

            if self.channel_seg_image is not None:
                    
                    self.update_mask = check_and_update_mask(self.mask, self.channel_seg_image)

            if self.seg_image is not None:
                    
                        self.update_mask = check_and_update_mask(self.mask, self.seg_image)
                      
            if self.seg_image is None and self.image is not None:
                        
                        self.update_mask = check_and_update_mask(self.mask, self.image)    
                           
            self.mask = self.update_mask.astype('uint16')
            self.timed_mask, self.boundary = boundary_points(self.mask, self.xcalibration, self.ycalibration, self.zcalibration)
        elif self.mask is None:
            if self.seg_image is not None:
                    
                    self.update_mask = np.zeros(self.seg_image.shape)
                        
            if self.seg_image is None and self.image is not None:

                    self.update_mask = np.zeros(self.image.shape) 
                            
            self.mask = self.update_mask.astype('uint16')
            self.mask[:,:,1:-1,1:-1] = 1
            self.timed_mask, self.boundary = boundary_points(self.mask, self.xcalibration, self.ycalibration, self.zcalibration)

          


    def _generate_generations(self, track):
         
        all_source_ids = []
        all_target_ids = [] 


        for edge in track.findall('Edge'):

                            source_id = int(edge.get(self.spot_source_id_key))
                            target_id = int(edge.get(self.spot_target_id_key))
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
                   
       
        #Get the leafs and splits     
        for target_id in all_target_ids:
             
             if target_id not in all_source_ids:
                  root_leaf.append(target_id)
             split_count = all_source_ids.count(target_id)
             if split_count > 1:
                      root_splits.append(target_id)

             
        #print('root and splits',root_root, root_leaf, root_splits)
        self._distance_root_leaf(root_root, root_leaf, root_splits)

        return root_root, root_splits, root_leaf


    def _iterate_split_down(self, root_root, root_leaf, root_splits):
         
         tracklet_count = str(0)
         self.assigned_tracket_counts = []
         for root_all in root_root:
                
                self.tracklet_dict[root_all] = str(tracklet_count)
                self.assigned_tracket_counts.append(tracklet_count)
                if root_all in self.edge_target_lookup:
                   target_cells = self.edge_target_lookup[root_all]
                   for i in range(len(target_cells)):
                        target_cell_id = target_cells[i]
                        if target_cell_id in root_splits:
                           tracklet_count = str(tracklet_count) + str(i) + str(1)
                           self._assign_tracklet_id(target_cell_id, root_splits, root_leaf, tracklet_count)  
                        if target_cell_id not in root_splits:
                           self._assign_tracklet_id(target_cell_id, root_splits, root_leaf, tracklet_count)    
                                     
         
    def _assign_tracklet_id(self, target_id, root_splits, root_leaf, tracklet_count ):
         
        if target_id in root_leaf:
               self.tracklet_dict[target_id] =  str(tracklet_count)
               self.assigned_tracket_counts.append(tracklet_count)
        if target_id not in root_leaf:  
            if target_id not in root_splits:
                            
                            self.tracklet_dict[target_id] = str(tracklet_count)
                            self.assigned_tracket_counts.append(tracklet_count)
                            if target_id in self.edge_target_lookup:
                                target_cells = self.edge_target_lookup[target_id]
                                
                                for i in range(len(target_cells)):
                                    target_cell_id = target_cells[i]
                                    self._assign_tracklet_id(target_cell_id, root_splits, root_leaf, tracklet_count )
            if target_id in root_splits:
                                    
                                    
                                    self.tracklet_dict[target_id] = str(tracklet_count)
                                    self.assigned_tracket_counts.append(tracklet_count)
                                    if target_id in self.edge_target_lookup:
                                        target_cells = self.edge_target_lookup[target_id]
                                        for i in range(len(target_cells)):
                                            target_cell_id = target_cells[i]
                                            if self.edge_source_lookup[target_cell_id] in root_splits:
                                                self._unique_split_id(target_cell_id, tracklet_count + str(i) + str(1))
                                            self._assign_tracklet_id(target_cell_id, root_splits, root_leaf,  tracklet_count + str(i) + str(1) )

                                    
  
         
    def _unique_split_id(self, target_id, tracklet_count):

        if tracklet_count not in self.assigned_tracket_counts:
               self.tracklet_dict[target_id] = str(tracklet_count)
               self.assigned_tracket_counts.append(tracklet_count)
        else:
        
            tracklet_count = str(tracklet_count) + str(1)
            self._unique_split_id(target_id, tracklet_count)


    def _distance_root_leaf(self, root_root, root_leaf, root_splits):


        
         gen_count = 0
         for root_all in root_root:
                self.generation_dict[root_all] = 0
                if root_all in self.edge_target_lookup:
                   target_cells = self.edge_target_lookup[root_all]
                   for i in range(len(target_cells)):
                        target_cell_id = target_cells[i]
                        if target_cell_id not in root_splits:
                               
                               self._recursive_path(target_cell_id, root_splits, root_leaf, gen_count )
                        if target_cell_id in root_splits:
                                       gen_count = gen_count + 1
                                       self._recursive_path(target_cell_id, root_splits, root_leaf, gen_count )

         
                              
    #Assign generation ID to each cell               
    def _recursive_path(self, target_id, root_splits, root_leaf, gen_count ):
         
        if target_id in root_leaf:
               self.generation_dict[target_id] =  gen_count
       
        if target_id not in root_leaf:  
            if target_id not in root_splits:
                            
                            self.generation_dict[target_id] = gen_count
                            if target_id in self.edge_target_lookup:
                                target_cells = self.edge_target_lookup[target_id]
                                for i in range(len(target_cells)):
                                    target_cell_id = target_cells[i]
                                    self._recursive_path(target_cell_id, root_splits, root_leaf, gen_count = gen_count)
            if target_id in root_splits:
                                    
                                    gen_count = gen_count + 1
                                    self.generation_dict[target_id] = gen_count
                                    if target_id in self.edge_target_lookup:
                                        target_cells = self.edge_target_lookup[target_id]
                                        for i in range(len(target_cells)):
                                            target_cell_id = target_cells[i]
                                            self._recursive_path(target_cell_id, root_splits, root_leaf, gen_count = gen_count)

                                    
                                    
                            
    def _get_boundary_dist(self, frame, testlocation, cellradius):
         
        if self.mask is not None:

                tree, indices, maskcentroid = self.timed_mask[str(int(float(frame)))]
                        
                # Get the location and distance to the nearest boundary point
                distance_cell_mask, locationindex = tree.query(testlocation)
                distance_cell_mask = max(0, distance_cell_mask - float(cellradius))
                   
        else:
                distance_cell_mask = 0
                maskcentroid = (1,1,1)

        return distance_cell_mask, maskcentroid        
         

    def _track_computer(self, track, track_id):
                             
                            
                            current_cell_ids = []
                            unique_tracklet_ids = []
                            
                            all_source_ids, all_target_ids =  self._generate_generations(track)
                            root_root, root_splits, root_leaf = self._create_generations(all_source_ids, all_target_ids) 
                            self._iterate_split_down(root_root, root_leaf, root_splits)
                            
                            number_dividing = len(root_splits)
                            # Determine if a track has divisions or none
                            if len(root_splits) > 0:
                                self.unique_track_mitosis_label[track_id] = [1, number_dividing]
                                dividing_trajectory = True
                                if int(track_id) not in self.AllTrackIds:
                                    self.AllTrackIds.append(int(track_id))
                                if int(track_id) not in self.DividingTrackIds:     
                                    self.DividingTrackIds.append(int(track_id))
                                        
                            else:
                                self.unique_track_mitosis_label[track_id] = [0, 0]
                                dividing_trajectory = False
                                if int(track_id) not in self.AllTrackIds:
                                    self.AllTrackIds.append(int(track_id))
                                if int(track_id) not in self.NormalTrackIds:    
                                    self.NormalTrackIds.append(int(track_id))

                            for leaf in root_leaf:
                                   source_leaf = self.edge_source_lookup[leaf]
                                   current_cell_ids.append(leaf) 
                                   self._dict_update(unique_tracklet_ids, leaf, track_id, source_leaf, None)
                                   self.unique_spot_properties[leaf].update({self.dividing_key : dividing_trajectory})
                                   self.unique_spot_properties[leaf].update({self.number_dividing_key : number_dividing})

                            for source_id in all_source_ids:
                                        target_ids = self.edge_target_lookup[source_id]
                                        current_cell_ids.append(source_id)
                                        #Root
                                        self.unique_spot_properties[source_id].update({self.dividing_key : dividing_trajectory})
                                        self.unique_spot_properties[source_id].update({self.number_dividing_key : number_dividing})
                                        if source_id not in all_target_ids:
                                                
                                                for target_id in target_ids:
                                                   self._dict_update(unique_tracklet_ids, source_id, track_id, None, target_id)
                                                   self.unique_spot_properties[target_id].update({self.dividing_key : dividing_trajectory})
                                                   self.unique_spot_properties[target_id].update({self.number_dividing_key : number_dividing})
                                        else:
                                              #Normal        
                                              source_source_id = self.edge_source_lookup[source_id]
                                              for target_id in target_ids:
                                                    self._dict_update(unique_tracklet_ids, source_id, track_id, source_source_id, target_id) 
                                                    self.unique_spot_properties[target_id].update({self.dividing_key : dividing_trajectory}) 
                                                    self.unique_spot_properties[target_id].update({self.number_dividing_key : number_dividing})
                                        

                                           
                            
                            for current_root in root_root:
                                   self.root_spots[int(current_root)] = self.unique_spot_properties[int(current_root)]
                            
                            self.all_current_cell_ids[int(track_id)] = current_cell_ids
                            for i in range(len(current_cell_ids)):
                                        
                                    k = int(current_cell_ids[i])    
                                    all_dict_values = self.unique_spot_properties[k]
                                   
                                    t = int(float(all_dict_values[self.frameid_key]))
                                    z = float(all_dict_values[self.zposid_key])
                                    y = float(all_dict_values[self.yposid_key])
                                    x = float(all_dict_values[self.xposid_key])
                                         

                                    spot_centroid = (round(z)/self.zcalibration, round(y)/self.ycalibration, round(x)/self.xcalibration)
                                    frame_spot_centroid = (t,round(z)/self.zcalibration, round(y)/self.ycalibration, round(x)/self.xcalibration)

                                    self.unique_spot_centroid[frame_spot_centroid] = k
                                    self.unique_track_centroid[frame_spot_centroid] = track_id

                                    if str(t) in self._timed_centroid:
                                           tree, spot_centroids = self._timed_centroid[str(t)]
                                           spot_centroids.append(spot_centroid)
                                           tree = spatial.cKDTree(spot_centroids)
                                           self._timed_centroid[str(t)] = tree, spot_centroids 
                                    else:
                                           spot_centroids = []
                                           spot_centroids.append(spot_centroid)
                                           tree = spatial.cKDTree(spot_centroids)
                                           self._timed_centroid[str(t)] = tree, spot_centroids
                            

    def _master_track_computer(self, track, track_id):
                             
                            
                            current_cell_ids = []
                           
                            
                            all_source_ids, all_target_ids =  self._generate_generations(track)
                            root_root, root_splits, root_leaf = self._create_generations(all_source_ids, all_target_ids) 
                            self._iterate_split_down(root_root, root_leaf, root_splits)
                            
                            # Determine if a track has divisions or none
                            number_dividing = len(root_splits)
                            if len(root_splits) > 0:
                                self.unique_track_mitosis_label[track_id] = [1, number_dividing]
                                dividing_trajectory = True
                                if int(track_id) not in self.AllTrackIds:
                                    self.AllTrackIds.append(int(track_id))
                                if int(track_id) not in self.DividingTrackIds:     
                                    self.DividingTrackIds.append(int(track_id))
                                        
                            else:
                                self.unique_track_mitosis_label[track_id] = [0, 0]
                                dividing_trajectory = False
                                if int(track_id) not in self.AllTrackIds:
                                    self.AllTrackIds.append(int(track_id))
                                if int(track_id) not in self.NormalTrackIds:    
                                    self.NormalTrackIds.append(int(track_id))

                            for leaf in root_leaf:
                                   self._second_channel_update(leaf, track_id)
                                   current_cell_ids.append(leaf) 
                                   self.unique_spot_properties[leaf].update({self.dividing_key : dividing_trajectory})
                                   self.unique_spot_properties[leaf].update({self.number_dividing_key : number_dividing})
                            for source_id in all_source_ids:
                                   self._second_channel_update(source_id, track_id)
                                   self.unique_spot_properties[source_id].update({self.dividing_key : dividing_trajectory})
                                   self.unique_spot_properties[source_id].update({self.number_dividing_key : number_dividing})
                                   current_cell_ids.append(source_id)
                                                

                            for current_root in root_root:
                                   self._second_channel_update(current_root, track_id)
                                   self.root_spots[int(current_root)] = self.unique_spot_properties[int(current_root)]
                                   self.unique_spot_properties[source_id].update({self.dividing_key : dividing_trajectory})
                                   self.unique_spot_properties[source_id].update({self.number_dividing_key : number_dividing})
                            
                            self.all_current_cell_ids[int(track_id)] = current_cell_ids
                           
                            for i in range(len(current_cell_ids)):
                                        
                                    k = int(current_cell_ids[i])   
                                    
                                    all_dict_values = self.unique_spot_properties[k]
                                    
                                  
                                    t = int(float(all_dict_values[self.frameid_key]))
                                    z = float(all_dict_values[self.zposid_key])
                                    y = float(all_dict_values[self.yposid_key])
                                    x = float(all_dict_values[self.xposid_key])
                                    

                                    frame_spot_centroid = (t,round(z)/self.zcalibration, round(y)/self.ycalibration, round(x)/self.xcalibration) 

                                    self.unique_spot_centroid[frame_spot_centroid] = k
                                    self.unique_track_centroid[frame_spot_centroid] = track_id
                                   
    def _second_channel_update(self, cell_id, track_id):
            
            if self.channel_seg_image is not None:
                
                frame = self.unique_spot_properties[int(cell_id)][self.frameid_key]
                z = self.unique_spot_properties[int(cell_id)][self.zposid_key]/self.zcalibration
                y = self.unique_spot_properties[int(cell_id)][self.yposid_key]/self.ycalibration
                x = self.unique_spot_properties[int(cell_id)][self.xposid_key]/self.xcalibration
                self._second_channel_spots(frame, z, y, x, cell_id, track_id)
        
    def _final_tracks(self, track_id):

                            current_cell_ids = self.all_current_cell_ids[int(track_id)]
                            current_tracklets = {}
                            current_tracklets_properties = {}
                            
                            for i in range(len(current_cell_ids)):
                                        
                                    k = int(current_cell_ids[i])    
                                    all_dict_values = self.unique_spot_properties[k]
                                    unique_id = str(all_dict_values[self.uniqueid_key])
                                    current_track_id = str(all_dict_values[self.trackid_key])
                                    t = int(float(all_dict_values[self.frameid_key]))
                                    z = float(all_dict_values[self.zposid_key])
                                    y = float(all_dict_values[self.yposid_key])
                                    x = float(all_dict_values[self.xposid_key])

                                    current_tracklets, current_tracklets_properties = self._tracklet_and_properties(all_dict_values, t, z, y, x, k, current_track_id, unique_id, current_tracklets, current_tracklets_properties)
                                    

                            current_tracklets = np.asarray(current_tracklets[str(track_id)])
                            current_tracklets_properties = np.asarray(current_tracklets_properties[str(track_id)])
                            
                            self.unique_tracks[track_id] = current_tracklets     
                            self.unique_track_properties[track_id] = current_tracklets_properties    

    def _tracklet_and_properties(self, all_dict_values, t, z, y, x, k, current_track_id, unique_id, current_tracklets, current_tracklets_properties):
           
                                    gen_id = int(float(all_dict_values[self.generationid_key]))
                                    speed = float(all_dict_values[self.speed_key])
                                    acceleration = float(all_dict_values[self.acceleration_key])
                                    motion_angle = float(all_dict_values[self.motion_angle_key])
                                    radial_angle = float(all_dict_values[self.radial_angle_key])
                                    radius = float(all_dict_values[self.radius_key])
                                    distance_cell_mask = float(all_dict_values[self.distance_cell_mask_key])
                                    total_intensity =  float(all_dict_values[self.total_intensity_key])
                                    volume_pixels = int(float(all_dict_values[self.quality_key]))

                                    

                                    if self.clusterclass_key in all_dict_values.keys():
                                           
                                           if all_dict_values[self.clusterclass_key] is not None:
                                                cluster_class = int(float(all_dict_values[self.clusterclass_key]))
                                                cluster_class_score = float(all_dict_values[self.clusterscore_key])
                                                eccentricity_comp_first = float(all_dict_values[self.eccentricity_comp_firstkey])
                                                eccentricity_comp_second = float(all_dict_values[self.eccentricity_comp_secondkey])
                                                surface_area = float(all_dict_values[self.surface_area_key])
                                                cell_axis_mask = float(all_dict_values[self.cellaxis_mask_key])
                                           else:
                                                cluster_class = None
                                                cluster_class_score = 0  
                                                eccentricity_comp_first = 0
                                                eccentricity_comp_second = 0 
                                                surface_area = 0
                                                cell_axis_mask = 0
                                               
                                    else:
                                            cluster_class = None
                                            cluster_class_score = 0  
                                            eccentricity_comp_first = 0
                                            eccentricity_comp_second = 0 
                                            surface_area = 0
                                            cell_axis_mask = 0      

                                    frame_spot_centroid = (t,round(z)/self.zcalibration, round(y)/self.ycalibration, round(x)/self.xcalibration) 
                                    self.unique_spot_centroid[frame_spot_centroid] = k

                                    if current_track_id in current_tracklets:
                                        tracklet_array = current_tracklets[current_track_id]
                                        current_tracklet_array = np.array([int(float(unique_id)), t, z/self.zcalibration, y/self.ycalibration, x/self.xcalibration])
                                        current_tracklets[current_track_id] = np.vstack((tracklet_array, current_tracklet_array))

                                        value_array = current_tracklets_properties[current_track_id]
                                        current_value_array = np.array([t, int(float(unique_id)), gen_id, radius, volume_pixels, eccentricity_comp_first, eccentricity_comp_second, surface_area, cluster_class, cluster_class_score, total_intensity, speed, motion_angle, acceleration, distance_cell_mask, radial_angle, cell_axis_mask])
                                        
                                        current_tracklets_properties[current_track_id] = np.vstack((value_array, current_value_array))

                                    else:
                                        current_tracklet_array = np.array([int(float(unique_id)), t, z/self.zcalibration, y/self.ycalibration, x/self.xcalibration])
                                        current_tracklets[current_track_id] = current_tracklet_array 

                                        current_value_array = np.array([t, int(float(unique_id)), gen_id, radius, volume_pixels,  eccentricity_comp_first, eccentricity_comp_second, surface_area, cluster_class, cluster_class_score, total_intensity, speed, motion_angle, acceleration, distance_cell_mask, radial_angle, cell_axis_mask ])
                                        current_tracklets_properties[current_track_id] = current_value_array

                                    return current_tracklets, current_tracklets_properties     

    def _master_spot_computer(self, frame):
          
          for Spotobject in frame.findall('Spot'):
                       
                        cell_id = int(Spotobject.get(self.spotid_key))
                      
                        if self.uniqueid_key in Spotobject.keys():
                        
                                        
                                self.unique_spot_properties[cell_id] = {
                                    self.cellid_key: int(float(Spotobject.get(self.spotid_key))), 
                                    self.frameid_key : int(float(Spotobject.get(self.frameid_key))),
                                    self.zposid_key : float(Spotobject.get(self.zposid_key)),
                                    self.yposid_key : float(Spotobject.get(self.yposid_key)),
                                    self.xposid_key : float(Spotobject.get(self.xposid_key)),
                                    self.total_intensity_key : (float(Spotobject.get(self.total_intensity_key))),
                                    self.mean_intensity_key : (float(Spotobject.get(self.mean_intensity_key))),
                                    self.radius_key : (float(Spotobject.get(self.radius_key))),
                                    self.quality_key : (float(Spotobject.get(self.quality_key))),
                                    self.distance_cell_mask_key: (float(Spotobject.get(self.distance_cell_mask_key))),
                                    self.uniqueid_key : str(Spotobject.get(self.uniqueid_key)),
                                    self.trackletid_key : str(Spotobject.get(self.trackletid_key)),
                                    self.generationid_key : str(Spotobject.get(self.generationid_key)),
                                    self.trackid_key : str(Spotobject.get(self.trackid_key)),
                                    self.motion_angle_key : (float(Spotobject.get(self.motion_angle_key))),
                                    self.speed_key : (float(Spotobject.get(self.speed_key))),
                                    self.acceleration_key : (float(Spotobject.get(self.acceleration_key))),
                                    self.radial_angle_key: float(Spotobject.get(self.radial_angle_key)),
                                }
                                if self.clusterclass_key in Spotobject.keys():
                                    if Spotobject.get(self.clusterclass_key) != 'None':
                                            self.unique_spot_properties[int(cell_id)].update({self.clusterclass_key : int(float(Spotobject.get(self.clusterclass_key))),
                                                                                        self.clusterscore_key : float(Spotobject.get(self.clusterscore_key)),
                                                                                        self.eccentricity_comp_firstkey : float(Spotobject.get(self.eccentricity_comp_firstkey)),
                                                                                        self.eccentricity_comp_secondkey : float(Spotobject.get(self.eccentricity_comp_secondkey)),
                                                                                        self.surface_area_key : float(Spotobject.get(self.surface_area_key)),
                                                                                        self.cellaxis_mask_key: float(Spotobject.get(self.cellaxis_mask_key))
                                                                                            })
                                            
                                            
                                    else:
                                            self.unique_spot_properties[int(cell_id)].update({self.clusterclass_key : None,
                                                                                        self.clusterscore_key : 0,
                                                                                        self.eccentricity_comp_firstkey : None,
                                                                                        self.eccentricity_comp_secondkey : None,
                                                                                        self.surface_area_key : None,
                                                                                        self.cellaxis_mask_key: None
                                                                                            }) 
                        elif self.uniqueid_key not in Spotobject.keys():
                                                          
                                            if self.detectorchannel == 1:
                                                    TOTAL_INTENSITY = Spotobject.get(self.total_intensity_ch2_key)
                                                    MEAN_INTENSITY = Spotobject.get(self.mean_intensity_ch2_key)
                                            else:        
                                                    TOTAL_INTENSITY = Spotobject.get(self.total_intensity_ch1_key)
                                                    MEAN_INTENSITY = Spotobject.get(self.mean_intensity_ch1_key)
                                            RADIUS = Spotobject.get(self.radius_key)
                                            QUALITY = Spotobject.get(self.quality_key)              
                                            self.unique_spot_properties[cell_id] = {
                                            self.cellid_key: int(cell_id), 
                                            self.frameid_key : int(float(Spotobject.get(self.frameid_key))),
                                            self.zposid_key : float(Spotobject.get(self.zposid_key)),
                                            self.yposid_key : float(Spotobject.get(self.yposid_key)),
                                            self.xposid_key : float(Spotobject.get(self.xposid_key)),
                                            
                                            self.total_intensity_key : (float(TOTAL_INTENSITY)),
                                            self.mean_intensity_key : (float(MEAN_INTENSITY)),
                                            self.radius_key : (float(RADIUS)),
                                            self.quality_key : (float(QUALITY))
                                        }
       
                   
                       
                     

    def _spot_computer(self, frame):

          for Spotobject in frame.findall('Spot'):
                        # Create object with unique cell ID
                        cell_id = int(Spotobject.get(self.spotid_key))
                        # Get the TZYX location of the cells in that frame
                        if self.detectorchannel == 1:
                                TOTAL_INTENSITY = Spotobject.get(self.total_intensity_ch2_key)
                                MEAN_INTENSITY = Spotobject.get(self.mean_intensity_ch2_key)
                        else:        
                                TOTAL_INTENSITY = Spotobject.get(self.total_intensity_ch1_key)
                                MEAN_INTENSITY = Spotobject.get(self.mean_intensity_ch1_key)
                        RADIUS = Spotobject.get(self.radius_key)
                        QUALITY = Spotobject.get(self.quality_key)
                        testlocation = (float(Spotobject.get(self.zposid_key)), float(Spotobject.get(self.yposid_key)),  float(Spotobject.get(self.xposid_key)))
                        frame = Spotobject.get(self.frameid_key)
                        distance_cell_mask, maskcentroid = self._get_boundary_dist(frame, testlocation, RADIUS)
                        
                        self.unique_spot_properties[cell_id] = {
                            self.cellid_key: int(cell_id), 
                            self.frameid_key : int(float(Spotobject.get(self.frameid_key))),
                            self.zposid_key : float(Spotobject.get(self.zposid_key)),
                            self.yposid_key : float(Spotobject.get(self.yposid_key)),
                            self.xposid_key : float(Spotobject.get(self.xposid_key)),
                            self.total_intensity_key : (float(TOTAL_INTENSITY)),
                            self.mean_intensity_key : (float(MEAN_INTENSITY)),
                            self.radius_key : (float(RADIUS)),
                            self.quality_key : (float(QUALITY)),
                            self.distance_cell_mask_key: float(distance_cell_mask),
                            self.maskcentroid_z_key: float(maskcentroid[0]),
                            self.maskcentroid_y_key: float(maskcentroid[1]),
                            self.maskcentroid_x_key: float(maskcentroid[2]) 
                        }
       
                        
            
                        
                                    

    
                        

    def _get_master_xml_data(self):
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
            self.detectorsettings = self.xml_content.find("Settings").find("DetectorSettings")
            self.basicsettings = self.xml_content.find("Settings").find("BasicSettings")
            self.detectorchannel = int(float(self.detectorsettings.get("TARGET_CHANNEL")))
            self.tstart = int(float(self.basicsettings.get("tstart")))
            self.tend = int(float(self.basicsettings.get("tend")))      

            print('Iterating over spots in frame')
            self.count = 0
            futures = []

            with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                
                for frame in self.Spotobjects.findall('SpotsInFrame'):
                            futures.append(executor.submit(self._master_spot_computer, frame))
                if self.progress_bar is not None:
                                
                                self.progress_bar.label = "Collecting Spots"
                                self.progress_bar.range = (
                                    0,
                                    len(futures),
                                )
                                self.progress_bar.show()

                for r in concurrent.futures.as_completed(futures):
                                self.count = self.count + 1
                                if self.progress_bar is not None:
                                    self.progress_bar.value =  self.count
                                r.result()    

            
            print(f'Iterating over tracks {len(self.filtered_track_ids)}')  
            self.count = 0
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                
                for track in self.tracks.findall('Track'):
                        
                        track_id = int(track.get(self.trackid_key))
                        if track_id in self.filtered_track_ids:
                                futures.append(executor.submit(self._master_track_computer, track, track_id))
                if self.progress_bar is not None:
                                
                                self.progress_bar.label = "Collecting Tracks"
                                self.progress_bar.range = (
                                    0,
                                    len(self.filtered_track_ids),
                                )
                                self.progress_bar.show()


                for r in concurrent.futures.as_completed(futures):
                                self.count = self.count + 1
                                if self.progress_bar is not None:
                                    self.progress_bar.value = self.count
                                r.result()
            
            if self.channel_seg_image is not None:  
                       
                       self._create_second_channel_xml()
                          

            for (k,v) in self.graph_split.items():
                           
                            daughter_track_id =  int(float(str(self.unique_spot_properties[int(float(k))][self.uniqueid_key])))
                            parent_track_id = int(float(str(self.unique_spot_properties[int(float(v))][self.uniqueid_key])))
                            self.graph_tracks[daughter_track_id] = parent_track_id

            print('getting attributes')                
            self._get_attributes()
           
            self.count = 0
            for track_id in self.filtered_track_ids:
                                    if self.progress_bar is not None:
                                        self.progress_bar.label = "Just one more thing"
                                        self.progress_bar.range = (
                                                0,
                                                len(self.filtered_track_ids),
                                            )
                                        self.progress_bar.show()
                                        self.count = self.count + 1
                                        self.progress_bar.value = self.count
                                    self._final_tracks(track_id) 

            if self.fourier:
                   print('computing Fourier')
                   self._compute_phenotypes()                        
            self._temporal_plots_trackmate()        


    def _create_second_channel_xml(self):
           
                    print('Transferring XML')   
                    channel_filtered_tracks = []            
                    for Spotobject in self.xml_root.iter('Spot'):
                            cell_id = int(Spotobject.get(self.spotid_key))
                            if cell_id in self.channel_unique_spot_properties.keys():        
                            
                                    new_positionx =  self.channel_unique_spot_properties[cell_id][self.xposid_key]
                                    new_positiony =  self.channel_unique_spot_properties[cell_id][self.yposid_key]
                                    new_positionz =  self.channel_unique_spot_properties[cell_id][self.zposid_key]

                                    new_total_intensity = self.channel_unique_spot_properties[cell_id][self.total_intensity_key]
                                    new_mean_intensity = self.channel_unique_spot_properties[cell_id][self.mean_intensity_key]

                                    new_radius = self.channel_unique_spot_properties[cell_id][self.radius_key]
                                    new_quality = self.channel_unique_spot_properties[cell_id][self.quality_key]
                                    new_distance_cell_mask = self.channel_unique_spot_properties[cell_id][self.distance_cell_mask_key]

                                    Spotobject.set(self.xposid_key, str(new_positionx))     
                                    Spotobject.set(self.yposid_key, str(new_positiony))
                                    Spotobject.set(self.zposid_key, str(new_positionz))

                                    Spotobject.set(self.total_intensity_key, str(new_total_intensity))     
                                    Spotobject.set(self.mean_intensity_key, str(new_mean_intensity))

                                    Spotobject.set(self.radius_key, str(new_radius))     
                                    Spotobject.set(self.quality_key, str(new_quality))
                                    Spotobject.set(self.distance_cell_mask_key, str(new_distance_cell_mask))
                                    track_id = self.channel_unique_spot_properties[int(cell_id)][self.trackid_key]
                                    channel_filtered_tracks.append(track_id)
                                    
                    self.xml_tree.write(os.path.join(self.channel_xml_path, self.channel_xml_name)) 

    def _get_xml_data(self):

                

                if self.channel_seg_image is not None:
                      self.channel_xml_content = self.xml_content
                      self.xml_tree = et.parse(self.xml_path)
                      self.xml_root = self.xml_tree.getroot()
                      self.channel_xml_name = 'second_channel_' + os.path.splitext(os.path.basename(self.xml_path))[0] + '.xml'
                      self.channel_xml_path = os.path.dirname(self.xml_path)
                      self._create_channel_tree()
                if self.cluster_model is not None and self.seg_image is not None:
                       self.master_xml_content = self.xml_content
                       self.master_xml_tree = et.parse(self.xml_path)
                       self.master_xml_root = self.master_xml_tree.getroot()
                       self.master_xml_name = 'master_' + self.master_extra_name  + os.path.splitext(os.path.basename(self.xml_path))[0] + '.xml'
                       self.master_xml_path = os.path.dirname(self.xml_path)      
                       
                self.unique_objects = {}
                self.unique_properties = {}
                self.AllTrackIds = []
                self.DividingTrackIds = []
                self.NormalTrackIds = []
                self.all_track_properties = []
                self.split_points_times = []

                
                
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
                self.detectorsettings = self.xml_content.find("Settings").find("DetectorSettings")
                self.basicsettings = self.xml_content.find("Settings").find("BasicSettings")
                self.detectorchannel = int(float(self.detectorsettings.get("TARGET_CHANNEL")))
                self.tstart = int(float(self.basicsettings.get("tstart")))
                self.tend = int(float(self.basicsettings.get("tend")))
                self._get_boundary_points()
                print('Iterating over spots in frame')
                self.count = 0
                futures = []

                with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                    
                    for frame in self.Spotobjects.findall('SpotsInFrame'):
                             futures.append(executor.submit(self._spot_computer, frame))
                    if self.progress_bar is not None:
                                 
                                    self.progress_bar.label = "Collecting Spots"
                                    self.progress_bar.range = (
                                        0,
                                        len(futures),
                                    )
                                    self.progress_bar.show()

                    for r in concurrent.futures.as_completed(futures):
                                    self.count = self.count + 1
                                    if self.progress_bar is not None:
                                      self.progress_bar.value =  self.count
                                    r.result()

                print(f'Iterating over tracks {len(self.filtered_track_ids)}')  
                self.count = 0
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                    
                    for track in self.tracks.findall('Track'):
                            
                            track_id = int(track.get(self.trackid_key))
                            if track_id in self.filtered_track_ids:
                                  futures.append(executor.submit(self._track_computer, track, track_id))
                    if self.progress_bar is not None:
                                 
                                    self.progress_bar.label = "Collecting Tracks"
                                    self.progress_bar.range = (
                                        0,
                                        len(self.filtered_track_ids),
                                    )
                                    self.progress_bar.show()


                    for r in concurrent.futures.as_completed(futures):
                                    self.count = self.count + 1
                                    if self.progress_bar is not None:
                                       self.progress_bar.value = self.count
                                    r.result()
                if self.channel_seg_image is not None:  
                       self._create_second_channel_xml()
                

                for (k,v) in self.graph_split.items():
                           
                            daughter_track_id =  int(float(str(self.unique_spot_properties[int(float(k))][self.uniqueid_key])))
                            parent_track_id = int(float(str(self.unique_spot_properties[int(float(v))][self.uniqueid_key])))
                            self.graph_tracks[daughter_track_id] = parent_track_id
                self._get_attributes()
                if self.cluster_model and self.seg_image is not None:
                       self._assign_cluster_class()
                       self._create_master_xml()
                self.count = 0 
                for track_id in self.filtered_track_ids:
                                    if self.progress_bar is not None:
                                        self.progress_bar.label = "Just one more thing"
                                        self.progress_bar.range = (
                                                0,
                                                len(self.filtered_track_ids),
                                            )
                                        self.progress_bar.show()
                                        self.count = self.count + 1
                                        self.progress_bar.value = self.count
                                    self._final_tracks(track_id) 

                if self.fourier:
                   print('computing Fourier')
                   self._compute_phenotypes()                        
                self._temporal_plots_trackmate()
                

    def _create_master_xml(self):
           
        
           for Spotobject in self.master_xml_root.iter('Spot'):
                                cell_id = int(Spotobject.get(self.spotid_key))
                                if cell_id in self.unique_spot_properties.keys():
                                       
                                       for k in self.unique_spot_properties[cell_id].keys():

                                           Spotobject.set(k, str(self.unique_spot_properties[cell_id][k]))   

                                      

           self.master_xml_tree.write(os.path.join(self.master_xml_path, self.master_xml_name))
           
                                       
                                       

    def _assign_cluster_class(self):
           
                    self.axes = self.axes.replace("T", "")
                    
                    for count, time_key in enumerate(self._timed_centroid.keys()):
                           
                           tree, spot_centroids = self._timed_centroid[time_key]
                           if self.progress_bar is not None:
                                self.progress_bar.label = "Computing clustering classes"
                                self.progress_bar.range = (
                                                                0,
                                                                len(self._timed_centroid.keys()) + 1,
                                                            )
                                self.progress_bar.value =  count 
                                self.progress_bar.show()

                           cluster_eval = Clustering(self.seg_image[int(time_key),:],  self.axes, self.num_points, self.cluster_model, key = time_key, progress_bar=self.progress_bar, batch_size = self.batch_size)       
                           cluster_eval._create_cluster_labels()
                           timed_cluster_label = cluster_eval.timed_cluster_label 
                           output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area = timed_cluster_label[time_key]
                           
                           for i in range(len(output_cluster_centroid)):
                                    centroid = output_cluster_centroid[i]
                                    cluster_class = output_cluster_class[i]
                                    cluster_score = output_cluster_score[i]
                                    quality = output_largest_eigenvalue[i]
                                    eccentricity_comp_firstyz = output_cloud_eccentricity[i]
                                    cell_axis = output_largest_eigenvector[i]
                                    surface_area = output_cloud_surface_area[i]
                                    dist, index = tree.query(centroid)
                                    closest_centroid = spot_centroids[index]
                                    frame_spot_centroid = (int(time_key),closest_centroid[0], closest_centroid[1], closest_centroid[2])
                                    closest_cell_id = self.unique_spot_centroid[frame_spot_centroid]
                                    mask_vector = [ float(self.unique_spot_properties[int(closest_cell_id)][self.maskcentroid_x_key]), float(self.unique_spot_properties[int(closest_cell_id)][self.maskcentroid_y_key]), float(self.unique_spot_properties[int(closest_cell_id)][self.maskcentroid_z_key]) ]
                                    cell_axis_mask = angular_change(cell_axis, mask_vector)
                                    
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.cellaxis_mask_key : cell_axis_mask})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.clusterclass_key : cluster_class})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.clusterscore_key : cluster_score})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.eccentricity_comp_firstkey : eccentricity_comp_firstyz[0]})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.eccentricity_comp_secondkey : eccentricity_comp_firstyz[1]})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.surface_area_key : surface_area})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.quality_key : quality})
                                    self.unique_spot_properties[int(closest_cell_id)].update({self.radius_key : quality * math.pow(self.zcalibration * self.xcalibration * self.ycalibration, 1.0/3.0) })


                                    
                           for (k,v) in self.root_spots.items():
                                  self.root_spots[k] = self.unique_spot_properties[k]         
                
    def _compute_phenotypes(self):

          for (k,v) in self.unique_tracks.items():
                
                track_id = k
                tracklet_properties = self.unique_track_properties[k]
                tracks = self.unique_tracks[k]
                Z = tracks[:,2]
                Y = tracks[:,3]
                X = tracks[:,4]
                time = tracklet_properties[:,0]
                unique_ids = tracklet_properties[:,1]
                unique_ids_set = set(unique_ids)
                generation_ids = tracklet_properties[:,2]
                radius = tracklet_properties[:,3]
                volume = tracklet_properties[:,4]
                eccentricity_comp_first = tracklet_properties[:,5]
                eccentricity_comp_second = tracklet_properties[:,6]
                surface_area = tracklet_properties[:,7]
                cluster_class = tracklet_properties[:,8]
                cluster_class_score = tracklet_properties[:,9]
                intensity = tracklet_properties[:,10]
                speed = tracklet_properties[:,11]
                motion_angle = tracklet_properties[:,12]
                acceleration = tracklet_properties[:,13]
                distance_cell_mask = tracklet_properties[:,14]
                radial_angle = tracklet_properties[:,15]
                cell_axis_mask = tracklet_properties[:,16]


                
                unique_fft_properties_tracklet = {}
                unique_cluster_properties_tracklet = {}
                self.unique_fft_properties[track_id] = {}
                self.unique_cluster_properties[track_id] = {}

                unique_shape_properties_tracklet = {}
                unique_dynamic_properties_tracklet = {}
                self.unique_shape_properties[track_id] = {}
                self.unique_dynamic_properties[track_id] = {}
                expanded_time = np.zeros(self.tend - self.tstart + 1)
                point_sample = expanded_time.shape[0]
                for i in range(len(expanded_time)):
                       expanded_time[i] = i 
                for current_unique_id in unique_ids_set:
                   
                   expanded_intensity = np.zeros(self.tend - self.tstart + 1)
                   
                   
                   current_time = []
                   current_z = []
                   current_y = []
                   current_x = []
                   current_intensity = []
                   current_cluster_class = []
                   current_cluster_class_score = []
                   current_radius = []
                   current_volume = []
                   current_speed = []
                   current_motion_angle = []
                   current_acceleration = []
                   current_distance_cell_mask = []
                   current_eccentricity_comp_first = []
                   current_eccentricity_comp_second = []
                   current_surface_area = []

                   current_radial_angle = []
                   current_cell_axis_mask = [] 
                   
                   for j in range(time.shape[0]):
                          if current_unique_id == unique_ids[j]:
                                 current_time.append(time[j])
                                 current_z.append(Z[j])
                                 current_y.append(Y[j])
                                 current_x.append(X[j])
                                 expanded_intensity[int(time[j])] = intensity[j]
                                 current_intensity.append(intensity[j])
                                 current_cluster_class.append(cluster_class[j])
                                 current_cluster_class_score.append(cluster_class_score[j])
                                 current_radius.append(radius[j])
                                 current_volume.append(volume[j])
                                 current_speed.append(speed[j])
                                 current_motion_angle.append(motion_angle[j])
                                 current_acceleration.append(acceleration[j])
                                 current_distance_cell_mask.append(distance_cell_mask[j])
                                 current_eccentricity_comp_first.append(eccentricity_comp_first[j])
                                 current_eccentricity_comp_second.append(eccentricity_comp_second[j])
                                 current_surface_area.append(surface_area[j])
                                 current_radial_angle.append(radial_angle[j])
                                 current_cell_axis_mask.append(cell_axis_mask[j])
                   current_time = np.asarray(current_time)
                   current_intensity = np.asarray(current_intensity)


                   current_cluster_class = np.asarray(current_cluster_class)
                   current_cluster_class_score = np.asarray(current_cluster_class_score)   

                   current_radius = np.asarray(current_radius)
                   current_volume = np.asarray(current_volume)
                   current_eccentricity_comp_first = np.asarray(current_eccentricity_comp_first)
                   current_eccentricity_comp_second = np.asarray(current_eccentricity_comp_second)
                   current_surface_area = np.asarray(current_surface_area)

                   current_speed = np.asarray(current_speed)
                   current_motion_angle = np.asarray(current_motion_angle)
                   current_acceleration = np.asarray(current_acceleration)
                   current_distance_cell_mask = np.asarray(current_distance_cell_mask)
                   current_radial_angle = np.asarray(current_radial_angle)
                   current_cell_axis_mask = np.asarray(current_cell_axis_mask)


                   
                   
                   if point_sample > 0:
                                xf_sample = fftfreq(point_sample, self.tcalibration)
                                fftstrip_sample = fft(expanded_intensity)
                                ffttotal_sample = np.abs(fftstrip_sample)
                                xf_sample = xf_sample[0 : len(xf_sample) // 2]
                                ffttotal_sample = ffttotal_sample[0 : len(ffttotal_sample) // 2]

                   unique_fft_properties_tracklet[current_unique_id] = expanded_time, expanded_intensity, xf_sample, ffttotal_sample
                   unique_cluster_properties_tracklet[current_unique_id] =  current_time, current_cluster_class, current_cluster_class_score
                   unique_shape_properties_tracklet[current_unique_id] = current_time, current_z, current_y, current_x, current_radius, current_volume, current_eccentricity_comp_first, current_eccentricity_comp_second, current_surface_area, current_cluster_class, current_cluster_class_score
                   unique_dynamic_properties_tracklet[current_unique_id] = current_time, current_speed, current_motion_angle, current_acceleration, current_distance_cell_mask, current_radial_angle, current_cell_axis_mask
                   self.unique_fft_properties[track_id].update({current_unique_id:unique_fft_properties_tracklet[current_unique_id]})
                   self.unique_cluster_properties[track_id].update({current_unique_id:unique_cluster_properties_tracklet[current_unique_id]})

                   self.unique_shape_properties[track_id].update({current_unique_id:unique_shape_properties_tracklet[current_unique_id]})
                   self.unique_dynamic_properties[track_id].update({current_unique_id:unique_dynamic_properties_tracklet[current_unique_id]})

    def _second_channel_spots(self, frame, z, y, x, cell_id, track_id):
           
            tree, centroids, labels, volume, intensity_mean, intensity_total, bounding_boxes = self._timed_channel_seg_image[str(int(float(frame)))]
            pixeltestlocation = (z,y,x)
            dist, index = tree.query(pixeltestlocation)


            bbox = bounding_boxes[index]
            sizez = abs(bbox[0] - bbox[3])
            sizey = abs(bbox[1] - bbox[4])
            sizex = abs(bbox[2] - bbox[5]) 
            veto_volume = sizex * sizey * sizez
            veto_radius = math.pow(3 * veto_volume / (4 * math.pi), 1.0 / 3.0)

            location = (centroids[index][0] * self.zcalibration, centroids[index][1]*self.ycalibration, centroids[index][2]*self.xcalibration)
            QUALITY = volume[index]
            RADIUS = math.pow(QUALITY, 1.0/3.0) * self.xcalibration * self.ycalibration * self.zcalibration
            distance_cell_mask, maskcentroid = self._get_boundary_dist(frame, location, RADIUS)
            if dist <= 2 * veto_radius:
                self.channel_unique_spot_properties[cell_id] = {
                        self.cellid_key: int(cell_id), 
                        self.frameid_key : int(frame),
                        self.zposid_key : float(centroids[index][0]* self.zcalibration),
                        self.yposid_key : float(centroids[index][1]* self.ycalibration),
                        self.xposid_key : float(centroids[index][2]* self.xcalibration),
                        self.trackid_key: int(track_id),
                        self.total_intensity_key : (float(intensity_total[index])),
                        self.mean_intensity_key : (float(intensity_mean[index])),

                        self.radius_key : (float(RADIUS)),
                        self.quality_key : (float(QUALITY)),
                        self.distance_cell_mask_key: float(distance_cell_mask),
                        self.maskcentroid_z_key: float(maskcentroid[0]),
                        self.maskcentroid_y_key: float(maskcentroid[1]),
                        self.maskcentroid_x_key: float(maskcentroid[2]) 

                }
            elif cell_id in self.edge_source_lookup.keys():
                    
                    self.channel_unique_spot_properties[cell_id] = self.unique_spot_properties[cell_id]

                    self.channel_unique_spot_properties[cell_id].update({self.total_intensity_key: None})
                    self.channel_unique_spot_properties[cell_id].update({self.mean_intensity_key: None})
                    self.channel_unique_spot_properties[cell_id].update({self.radius_key: None})
                    self.channel_unique_spot_properties[cell_id].update({self.quality_key: None})


                                 
    def _dict_update(self, unique_tracklet_ids: List,  cell_id: int, track_id: int, source_id: int, target_id: int):

 
        generation_id = self.generation_dict[cell_id]
        tracklet_id = self.tracklet_dict[cell_id]

        unique_id = str(track_id) + str(self.max_track_id) + str(generation_id) + str(tracklet_id)
        
        vec_mask = [float(self.unique_spot_properties[int(cell_id)][self.maskcentroid_x_key]), float(self.unique_spot_properties[int(cell_id)][self.maskcentroid_y_key]), float(self.unique_spot_properties[int(cell_id)][self.maskcentroid_z_key]) ]

        vec_cell = [float(self.unique_spot_properties[int(cell_id)][self.xposid_key]) , 
                            float(self.unique_spot_properties[int(cell_id)][self.yposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.zposid_key])]

        angle = angular_change(vec_mask, vec_cell)

        self.unique_spot_properties[int(cell_id)].update({self.radial_angle_key : angle})                    

        unique_tracklet_ids.append(str(unique_id))
        self.unique_spot_properties[int(cell_id)].update({self.clusterclass_key : None})
        self.unique_spot_properties[int(cell_id)].update({self.clusterscore_key : 0})
        self.unique_spot_properties[int(cell_id)].update({self.uniqueid_key : str(unique_id)})
        self.unique_spot_properties[int(cell_id)].update({self.trackletid_key : str(tracklet_id)}) 
        self.unique_spot_properties[int(cell_id)].update({self.generationid_key : str(generation_id)}) 
        self.unique_spot_properties[int(cell_id)].update({self.trackid_key : str(track_id)})
        self.unique_spot_properties[int(cell_id)].update({self.motion_angle_key : 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.speed_key : 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.acceleration_key : 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.eccentricity_comp_firstkey : None})
        self.unique_spot_properties[int(cell_id)].update({self.eccentricity_comp_secondkey : None})
        self.unique_spot_properties[int(cell_id)].update({self.surface_area_key : None})
        self.unique_spot_properties[int(cell_id)].update({self.cellaxis_mask_key : None})

        if source_id is not None:
            self.unique_spot_properties[int(cell_id)].update({self.beforeid_key : int(source_id)})
            vec_1 = [float(self.unique_spot_properties[int(cell_id)][self.xposid_key]) - float(self.unique_spot_properties[int(source_id)][self.xposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.yposid_key]) - float(self.unique_spot_properties[int(source_id)][self.yposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.zposid_key]) -  float(self.unique_spot_properties[int(source_id)][self.zposid_key])]
            speed = np.sqrt(np.dot(vec_1, vec_1))/self.tcalibration
            self.unique_spot_properties[int(cell_id)].update({self.speed_key : speed})

            motion_angle = angular_change(vec_mask, vec_1)

            self.unique_spot_properties[int(cell_id)].update({self.motion_angle_key : motion_angle}) 

            if source_id in self.edge_source_lookup:
                    pre_source_id = self.edge_source_lookup[source_id]
                    
                    vec_2 = [float(self.unique_spot_properties[int(cell_id)][self.xposid_key]) - 2 * float(self.unique_spot_properties[int(source_id)][self.xposid_key]) + float(self.unique_spot_properties[int(pre_source_id)][self.xposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.yposid_key]) - 2 * float(self.unique_spot_properties[int(source_id)][self.yposid_key]) + float(self.unique_spot_properties[int(pre_source_id)][self.yposid_key]), 
                            float(self.unique_spot_properties[int(cell_id)][self.zposid_key]) -  2 * float(self.unique_spot_properties[int(source_id)][self.zposid_key]) + float(self.unique_spot_properties[int(pre_source_id)][self.zposid_key])]
                    acc = np.sqrt(np.dot(vec_2, vec_2))/self.tcalibration
                    
                    self.unique_spot_properties[int(cell_id)].update({self.acceleration_key : acc})
        elif source_id is None:
            self.unique_spot_properties[int(cell_id)].update({self.beforeid_key : None}) 
            

        if target_id is not None:       
            self.unique_spot_properties[int(cell_id)].update({self.afterid_key : int(target_id)}) 
        elif target_id is None:
            self.unique_spot_properties[int(cell_id)].update({self.afterid_key : None})
            
        self._second_channel_update(cell_id, track_id)    


    def _temporal_plots_trackmate(self):
    
    
    
                self.Attr = {}
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

                self.mitotic_mean_acc = []
                self.mitotic_var_acc = []

                self.mitotic_mean_directional_change = []
                self.mitotic_var_directional_change = []

                self.mitotic_mean_distance_cell_mask = []
                self.mitotic_var_distance_cell_mask = []

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

                self.non_mitotic_mean_acc = []
                self.non_mitotic_var_acc = []

                self.non_mitotic_mean_directional_change = []
                self.non_mitotic_var_directional_change = []

                self.non_mitotic_mean_distance_cell_mask = []
                self.non_mitotic_var_distance_cell_mask = []

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

                self.all_mean_acc = []
                self.all_var_acc = []

                self.all_mean_directional_change = []
                self.all_var_directional_change = []

                self.all_mean_distance_cell_mask = []
                self.all_var_distance_cell_mask = []

                self.mitotic_cluster_class = []
                self.non_mitotic_cluster_class = []
                self.all_cluster_class = []

                all_spots_tracks = {}
                for (k,v) in self.unique_spot_properties.items():
                      
                      all_spots = self.unique_spot_properties[k]
                      if self.trackid_key in all_spots:
                          all_spots_tracks[k] = all_spots
                

                
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                      
                    for i in tqdm(range(starttime, endtime), total=endtime - starttime):
                      
                        futures.append(executor.submit(self._compute_temporal, i, all_spots_tracks))
 
                    [r.result() for r in concurrent.futures.as_completed(futures)]


    def _compute_temporal(self, i, all_spots_tracks):                
                    mitotic_disp_z = []
                    mitotic_disp_y = []
                    mitotic_disp_x = []
                    mitotic_radius = []
                    mitotic_speed = []
                    mitotic_acc = []
                    mitotic_directional_change = []
                    mitotic_cluster_class = []
                    mitotic_distance_cell_mask = []

                    non_mitotic_disp_z = []
                    non_mitotic_disp_y = []
                    non_mitotic_disp_x = []
                    non_mitotic_radius = []
                    non_mitotic_speed = []
                    non_mitotic_acc = []
                    non_mitotic_directional_change = []
                    non_mitotic_cluster_class = []
                    non_mitotic_distance_cell_mask = []
                    
                    all_disp_z = []
                    all_disp_y = []
                    all_disp_x = []
                    all_radius = []
                    all_speed = []
                    all_acc = []
                    all_directional_change = []
                    all_cluster_class = []
                    all_distance_cell_mask = []




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
                                        mitotic_acc.append(all_spots_tracks[k][self.acceleration_key])
                                        mitotic_directional_change.append(all_spots_tracks[k][self.motion_angle_key])
                                        mitotic_distance_cell_mask.append(all_spots_tracks[k][self.distance_cell_mask_key])
                                        if self.clusterclass_key in all_spots_tracks[k].keys() :
                                               mitotic_cluster_class.append(all_spots_tracks[k][self.clusterclass_key])


                                  if not mitotic:
                                        non_mitotic_disp_z.append(all_spots_tracks[k][self.zposid_key])
                                        non_mitotic_disp_y.append(all_spots_tracks[k][self.yposid_key])
                                        non_mitotic_disp_x.append(all_spots_tracks[k][self.xposid_key])
                                        non_mitotic_radius.append(all_spots_tracks[k][self.radius_key])
                                        non_mitotic_speed.append(all_spots_tracks[k][self.speed_key])
                                        non_mitotic_acc.append(all_spots_tracks[k][self.acceleration_key])
                                        non_mitotic_directional_change.append(all_spots_tracks[k][self.motion_angle_key])
                                        non_mitotic_distance_cell_mask.append(all_spots_tracks[k][self.distance_cell_mask_key])
                                        if self.clusterclass_key in all_spots_tracks[k].keys() :
                                               non_mitotic_cluster_class.append(all_spots_tracks[k][self.clusterclass_key])

                                  all_disp_z.append(all_spots_tracks[k][self.zposid_key])
                                  all_disp_y.append(all_spots_tracks[k][self.yposid_key])
                                  all_disp_x.append(all_spots_tracks[k][self.xposid_key])
                                  all_radius.append(all_spots_tracks[k][self.radius_key])
                                  all_speed.append(all_spots_tracks[k][self.speed_key])
                                  all_acc.append(all_spots_tracks[k][self.acceleration_key])
                                  all_directional_change.append(all_spots_tracks[k][self.motion_angle_key])   
                                  all_distance_cell_mask.append(all_spots_tracks[k][self.distance_cell_mask_key])
                                  if self.clusterclass_key in all_spots_tracks[k].keys() :
                                               all_cluster_class.append(all_spots_tracks[k][self.clusterclass_key])    
                                              

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

                    self.mitotic_cluster_class.append(np.asarray(mitotic_cluster_class))
                    self.non_mitotic_cluster_class.append(np.asarray(non_mitotic_cluster_class))
                    self.all_cluster_class.append(np.asarray(all_cluster_class))

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

                    self.mitotic_mean_acc.append(np.mean(mitotic_acc))
                    self.mitotic_var_acc.append(np.std(mitotic_acc))

                    self.mitotic_mean_directional_change.append(np.mean(mitotic_directional_change))
                    self.mitotic_var_directional_change.append(np.std(mitotic_directional_change))

                    self.mitotic_mean_distance_cell_mask.append(np.mean(mitotic_distance_cell_mask))
                    self.mitotic_var_distance_cell_mask.append(np.std(mitotic_distance_cell_mask))

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

                    self.non_mitotic_mean_acc.append(np.mean(non_mitotic_acc))
                    self.non_mitotic_var_acc.append(np.std(non_mitotic_acc))

                    self.non_mitotic_mean_directional_change.append(np.mean(non_mitotic_directional_change))
                    self.non_mitotic_var_directional_change.append(np.std(non_mitotic_directional_change)) 

                    self.non_mitotic_mean_distance_cell_mask.append(np.mean(non_mitotic_distance_cell_mask))
                    self.non_mitotic_var_distance_cell_mask.append(np.std(non_mitotic_distance_cell_mask))


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

                    self.all_mean_acc.append(np.mean(all_acc))
                    self.all_var_acc.append(np.std(all_acc))



                    self.all_mean_directional_change.append(np.mean(all_directional_change))
                    self.all_var_directional_change.append(np.std(all_directional_change))

                    self.all_mean_distance_cell_mask.append(np.mean(all_distance_cell_mask))
                    self.all_var_distance_cell_mask.append(np.std(all_distance_cell_mask))
                            
        
def boundary_points(mask, xcalibration, ycalibration, zcalibration):

    ndim = len(mask.shape)
    timed_mask = {}
    mask = mask > 0
    mask = mask.astype('uint8')
    # YX shaped object
    if ndim == 2:
        
        boundary = find_boundaries(mask)
        regioncentroid = (0,) + compute_centroid(boundary) 
        indices = np.where(boundary > 0)
        real_indices = np.transpose(np.asarray(indices)).copy()

        for j in range(0, len(real_indices)):

            real_indices[j][0] = real_indices[j][0] * ycalibration
            real_indices[j][1] = real_indices[j][1] * xcalibration

        tree = spatial.cKDTree(real_indices)
        # This object contains list of all the points for all the labels in the Mask image with the label id and volume of each label
        timed_mask[str(0)] = [tree, indices, regioncentroid]

    # TYX shaped object
    if ndim == 3:


        for i in tqdm(range(0, mask.shape[0])):
                
                boundary = find_boundaries(mask[i,:])
                regioncentroid = (0,) + compute_centroid(boundary) 
                indices = np.where(boundary > 0)
                real_indices = np.transpose(np.asarray(indices)).copy()

                for j in range(0, len(real_indices)):

                    real_indices[j][0] = real_indices[j][0] * ycalibration
                    real_indices[j][1] = real_indices[j][1] * xcalibration

                tree = spatial.cKDTree(real_indices)

                timed_mask[str(i)] = [tree, indices, regioncentroid]
            
    # TZYX shaped object
    if ndim == 4:
        print('Masks made into a 4D cylinder, up')
        boundary = np.zeros(
            [mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]]
        )
        for i in range(0, mask.shape[0]):
            
            boundary[i,:] = find_boundaries(mask[i,:])
            regioncentroid = compute_centroid(boundary[i,:]) 
            indices = np.where(boundary[i,:] > 0)
            real_indices = np.transpose(np.asarray(indices)).copy()

            for j in range(0, len(real_indices)):

                    real_indices[j][0] = real_indices[j][0] * zcalibration
                    real_indices[j][1] = real_indices[j][1] * ycalibration
                    real_indices[j][2] = real_indices[j][2] * xcalibration

            tree = spatial.cKDTree(real_indices)
            timed_mask[str(i)] = [tree, indices, regioncentroid]
    print('Computed the boundary points')

    return timed_mask, boundary        

def compute_centroid(binary_image):
    # Ensure binary image is a NumPy array
    binary_image = np.array(binary_image)

    white_pixels = np.where(binary_image == 1)
    num_pixels = len(white_pixels[0])

    # Compute the centroid of the white pixels in the boundary image
    centroid = np.zeros(binary_image.ndim)
    for dim in range(binary_image.ndim):
        centroid[dim] = white_pixels[dim].sum() / num_pixels

    return centroid



 

def get_csv_data(csv):

        dataset = pd.read_csv(
            csv, delimiter=",", encoding="unicode_escape", low_memory=False
        )[3:]
        dataset_index = dataset.index
        return dataset, dataset_index
    
def get_spot_dataset(spot_dataset, track_analysis_spot_keys, xcalibration, ycalibration, zcalibration, AttributeBoxname, detectionchannel):
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
        

        ignore_values = [track_analysis_spot_keys["mean_intensity"],track_analysis_spot_keys["total_intensity"]] 
        for (k,v) in track_analysis_spot_keys.items():

                if detectionchannel == 1:
                     if k == "mean_intensity_ch2":
                           value = track_analysis_spot_keys["mean_intensity"]
                           AllValues[value] = spot_dataset[v].astype("float")
                     if k == "total_intensity_ch2":
                           value = track_analysis_spot_keys["total_intensity"]
                           AllValues[value] = spot_dataset[v].astype("float")       

                if v not in ignore_values:
                       
                       AllValues[v] = spot_dataset[v].astype("float")

        AllValues[posix] = round(LocationX,3)
        AllValues[posiy] = round(LocationY,3)
        AllValues[posiz] = round(LocationZ,3)
        AllValues[frame] = round(LocationT,3)
        Attributeids = []
        Attributeids.append(AttributeBoxname)
        for attributename in AllValues.keys():
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
    


def prob_sigmoid(x):
    return 1 - math.exp(-x)


def angular_change(vec_0, vec_1):
        
        vec_0 = vec_0 / np.linalg.norm(vec_0)
        vec_1 = vec_1 / np.linalg.norm(vec_1)
        angle = np.arccos(np.clip(np.dot(vec_0, vec_1), -1.0, 1.0))
        angle = angle * 180 / np.pi
        return angle
     

def eval_bool(value):
                  
        if value  == 'True': 
                div_key = True
        else:
                div_key = False 

        return div_key                

def check_and_update_mask(mask,image):
       
        if len(mask.shape) < len(image.shape):
            update_mask = np.zeros(
                            [
                                image.shape[0],
                                image.shape[1],
                                image.shape[2],
                                image.shape[3],
                            ]
                        )
            for i in range(0, update_mask.shape[0]):
                for j in range(0, update_mask.shape[1]):

                    update_mask[i, j, :, :] = mask[i, :, :]
        else:
                update_mask = mask

        return update_mask        
       

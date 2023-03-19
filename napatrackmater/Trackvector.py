from . import TrackMate
from pathlib import Path 
import lxml.etree as et
import concurrent
import os
import numpy as np
import napari

class TrackVector(TrackMate):
       
        def __init__(self, viewer, image, master_xml_path: Path, spot_csv_path: Path, track_csv_path: Path, edges_csv_path: Path, t_minus: int = 0, t_plus: int = 10, x_start : int = 0, x_end: int = 10,
                    y_start: int = 0, y_end: int = 10, show_tracks: bool = True):
              
              

              super().__init__(None,  spot_csv_path, track_csv_path, edges_csv_path, image = image, AttributeBoxname = "AttributeIDBox", TrackAttributeBoxname = "TrackAttributeIDBox", TrackidBox = "All", axes = 'TZYX', master_xml_path = None )
              self._viewer = viewer
              self._image = image
              self.master_xml_path = master_xml_path
              self.spot_csv_path = spot_csv_path
              self.track_csv_path = track_csv_path
              self.edges_csv_path = edges_csv_path
              self._t_minus = t_minus
              self._t_plus = t_plus 
              self._x_start = x_start 
              self._x_end = x_end 
              self._y_start = y_start 
              self._y_end = y_end 
              self._show_tracks = show_tracks
              xml_parser = et.XMLParser(huge_tree=True)
               
              

              self.unique_morphology_dynamic_properties = {}
              self.unique_mitosis_label = {}
              self.non_unique_mitosis_label = {}  
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

                        self._get_track_vector_xml_data()

        @property
        def viewer(self):
               return self._viewer 
        
        @viewer.setter
        def viewer(self, value):
               self._viewer = value 


        @property
        def x_start(self):
               return self._x_start
        
        @x_start.setter
        def x_start(self, value):
               self._x_start = value

        @property
        def y_start(self):
               return self._y_start
        
        @y_start.setter
        def y_start(self, value):
               self._y_start = value

        @property
        def x_end(self):
               return self._x_end
        
        @x_end.setter
        def x_end(self, value):
               self._x_end = value

        @property
        def y_end(self):
               return self._y_end
        
        @y_end.setter
        def y_end(self, value):
               self._y_end = value       

       

        @property
        def t_minus(self):
               return self._t_minus 
        
        @t_minus.setter
        def t_minus(self, value):
               self._t_minus = value

        @property
        def t_plus(self):
               return self._t_plus 
        
        @t_plus.setter
        def t_plus(self, value):
               self._t_plus = value 

        @property 
        def show_tracks(self):
               return self._show_tracks

        @show_tracks.setter
        def show_tracks(self, value):
               self._show_tracks = value             







        def _get_track_vector_xml_data(self):
            
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
            self.xmin = int(float(self.basicsettings.get("xstart")))
            self.xmax = int(float(self.basicsettings.get("xend")))      
            self.ymin = int(float(self.basicsettings.get("ystart")))
            self.ymax = int(float(self.basicsettings.get("yend")))

            if self.x_end > self.xmax:
                    self.x_end = self.xmax
            if self.y_end > self.ymax:
                    self.y_end = self.ymax

            if self.x_start < self.xmin:
                    self.x_start = self.xmin
            if self.y_start < self.ymin:
                    self.y_start = self.ymin                         
            print('Iterating over spots in frame')
            self.count = 0
            futures = []

            with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                
                for frame in self.Spotobjects.findall('SpotsInFrame'):
                            futures.append(executor.submit(self._master_spot_computer, frame))
                
                [r.result() for r in concurrent.futures.as_completed(futures)]
                   

            
            print(f'Iterating over tracks {len(self.filtered_track_ids)}')  
            self.count = 0
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                
                for track in self.tracks.findall('Track'):
                        
                        track_id = int(track.get(self.trackid_key))
                        if track_id in self.filtered_track_ids:
                                futures.append(executor.submit(self._master_track_computer, track, track_id))

                [r.result() for r in concurrent.futures.as_completed(futures)]
            

            print('getting attributes')                
            self._get_attributes()
           
            self.count = 0
             

            self._compute_cluster_phenotypes()                        

        def _interactive_function(self):
               
               self.unique_tracks = {}
               self.unique_track_properties = {}
               for track_id in self.filtered_track_ids:
                                    
                                    self._final_morphological_dynamic_vectors(track_id)

               if self._show_tracks:
                        
                       
                        if len(list(self._viewer.layers)) > 0:
                            layer_types = []
                            for layer in list(self._viewer.layers):
                                   layer_types.append(type(layer))

                            if  napari.layers.Image not in layer_types:   
                                      self._viewer.add_image(self._image)
                        else:

                               self._viewer.add_image(self._image)              


                        if len(self.unique_tracks.keys()) > 0:   
                                unique_tracks = np.concatenate(
                                    [
                                        self.unique_tracks[unique_track_id]
                                        for unique_track_id in self.unique_tracks.keys()
                                    ]
                                )     

                                unique_tracks_properties = np.concatenate(
                                   [
                                   self.unique_track_properties[unique_track_id]
                                   for unique_track_id in self.unique_track_properties.keys()
                                   ]
                            )             

                                features = {
                                          "time": np.asarray(unique_tracks_properties, dtype="float64")[:, 0],
                                          
                                          "generation": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 2],
                                          
                                          "speed": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 3],
                                          
                                          "directional_change_rate": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 4],
                                          
                                          "total-intensity": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 5],
                                          
                                          "volume_pixels": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 6],
                                          
                                          "acceleration": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 7],
                                   
                                          "cluster_class": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 8],
                                          
                                          "cluster_score": 
                                                 np.asarray(unique_tracks_properties, dtype="float64")[:, 9],
                                          
                                          }   
                                for layer in list(self._viewer.layers):
                                   if (
                                          "Track" == layer.name
                                          or "Boxes" == layer.name
                                          or "Track_points" == layer.name
                                   ):
                                          self._viewer.layers.remove(layer)
                                   vertices = unique_tracks[:, 1:]
                                self._viewer.add_points(vertices, name="Track_points", size=1)
                                self._viewer.add_tracks(
                                unique_tracks,
                                name="Track",
                                features = features
                            )
                                


        def _final_morphological_dynamic_vectors(self, track_id):
                
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
                        if t >= self._t_minus and t <=  self._t_plus and x >= self._x_start and x <= self._x_end and y >= self._y_start and y <= self._y_end:
                                gen_id = int(float(all_dict_values[self.generationid_key]))
                                speed = float(all_dict_values[self.speed_key])
                                acceleration = float(all_dict_values[self.acceleration_key])
                                dcr = float(all_dict_values[self.directional_change_rate_key])
                                radius = float(all_dict_values[self.radius_key])
                               
                                total_intensity =  float(all_dict_values[self.total_intensity_key])
                                volume_pixels = int(float(all_dict_values[self.quality_key]))
                                if self.clusterclass_key in all_dict_values.keys():
                                        
                                        if all_dict_values[self.clusterclass_key] is not None:
                                            cluster_class = int(float(all_dict_values[self.clusterclass_key]))
                                            cluster_class_score = float(all_dict_values[self.clusterscore_key])
                                        else:
                                            cluster_class = None
                                            cluster_class_score = 0     
                                else:
                                        cluster_class = None
                                        cluster_class_score = 0       

                                spot_centroid = (round(z)/self.zcalibration, round(y)/self.ycalibration, round(x)/self.xcalibration)

                                self.unique_spot_centroid[spot_centroid] = k

                                if current_track_id in current_tracklets:
                                    tracklet_array = current_tracklets[current_track_id]
                                    current_tracklet_array = np.array([int(float(unique_id)), t, z/self.zcalibration, y/self.ycalibration, x/self.xcalibration])
                                    current_tracklets[current_track_id] = np.vstack((tracklet_array, current_tracklet_array))

                                    value_array = current_tracklets_properties[current_track_id]
                                    current_value_array = np.array([t, int(float(unique_id)), gen_id, speed, dcr, total_intensity, volume_pixels, acceleration, cluster_class, cluster_class_score])
                                    current_tracklets_properties[current_track_id] = np.vstack((value_array, current_value_array))

                                else:
                                    current_tracklet_array = np.array([int(float(unique_id)), t, z/self.zcalibration, y/self.ycalibration, x/self.xcalibration])
                                    current_tracklets[current_track_id] = current_tracklet_array 

                                    current_value_array = np.array([t, int(float(unique_id)), gen_id, speed, dcr, total_intensity, volume_pixels, acceleration, cluster_class, cluster_class_score])
                                    current_tracklets_properties[current_track_id] = current_value_array

                if str(track_id) in current_tracklets:
                        current_tracklets = np.asarray(current_tracklets[str(track_id)])
                        current_tracklets_properties = np.asarray(current_tracklets_properties[str(track_id)])
                        if len(current_tracklets.shape) == 2:
                            self.unique_tracks[track_id] = current_tracklets     
                            self.unique_track_properties[track_id] = current_tracklets_properties 

                
            
                        

        def _compute_cluster_phenotypes(self):
                
            for (k,v) in self.unique_tracks.items():
                
                track_id = k
                tracklet_properties = self.unique_track_properties[k] 
                intensity = tracklet_properties[:,5]
                time = tracklet_properties[:,0]
                unique_ids = tracklet_properties[:,1]
                unique_ids_set = set(unique_ids)
                cluster_class_score = tracklet_properties[:,9]
                cluster_class = tracklet_properties[:,8]
                
                
                unique_cluster_properties_tracklet = {}
                self.unique_cluster_properties[track_id] = {}
                for current_unique_id in unique_ids_set:
                   
                   current_time = []
                   current_cluster_class = []
                   current_cluster_class_score = []
                   for j in range(time.shape[0]):
                          if current_unique_id == unique_ids[j]:
                                 current_time.append(time[j])
                                 current_intensity.append(intensity[j])
                                 current_cluster_class.append(cluster_class[j])
                                 current_cluster_class_score.append(cluster_class_score[j])
                   current_time = np.asarray(current_time)
                   current_intensity = np.asarray(current_intensity)
                   current_cluster_class = np.asarray(current_cluster_class)
                   current_cluster_class_score = np.asarray(current_cluster_class_score)               
                 

                   unique_cluster_properties_tracklet[current_unique_id] =  current_time, current_cluster_class, current_cluster_class_score
                   self.unique_cluster_properties[track_id].update({current_unique_id:unique_cluster_properties_tracklet[current_unique_id]})



                



               
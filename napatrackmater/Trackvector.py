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
         
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
                
                for track in self.tracks.findall('Track'):
                        
                        track_id = int(track.get(self.trackid_key))
                        if track_id in self.filtered_track_ids:
                                futures.append(executor.submit(self._master_track_computer, track, track_id))

                [r.result() for r in concurrent.futures.as_completed(futures)]
            

            print('getting attributes')                
            self._get_attributes()
                                    
        def _compute_track_vectors(self):


               self.current_shape_dynamic_vectors = []
               for k in self.unique_dynamic_properties.keys():
                      dividing, number_dividing = self.unique_track_mitosis_label[k]
                      nested_unique_dynamic_properties = self.unique_dynamic_properties[k]
                      nested_unique_shape_properties = self.unique_shape_properties[k]
                      for current_unique_id in nested_unique_dynamic_properties.keys():
                             
                             unique_dynamic_properties_tracklet = nested_unique_dynamic_properties[current_unique_id]
                             current_time, speed, motion_angle, acceleration, distance_cell_mask, radial_angle, cell_axis_mask = unique_dynamic_properties_tracklet
                             unique_shape_properties_tracklet = nested_unique_shape_properties[current_unique_id]
                             current_time, current_z, current_y, current_x, radius, volume, eccentricity_comp_first, eccentricity_comp_second, surface_area, current_cluster_class, current_cluster_class_score = unique_shape_properties_tracklet
                             
                             track_id_array = np.ones(current_time.shape)
                             dividing_array = np.ones(current_time.shape)
                             number_dividing_array = np.ones(current_time.shape)
                             for i in range(track_id_array.shape[0]):
                                    track_id_array[i] = track_id_array[i] * current_unique_id
                                    dividing_array[i] = dividing_array[i] * dividing 
                                    number_dividing_array[i] = number_dividing_array[i] * number_dividing
                             self.current_shape_dynamic_vectors.append([ track_id_array, current_time, current_z, current_y, current_x, dividing_array, number_dividing_array, radius, volume, eccentricity_comp_first, eccentricity_comp_second, surface_area, current_cluster_class, speed, motion_angle, acceleration, distance_cell_mask, radial_angle, cell_axis_mask])

          
               print(f'returning shape and dynamic vectors as list {len(self.current_shape_dynamic_vectors)}')


        def _interactive_function(self):
               
               self.unique_tracks = {}
               self.unique_track_properties = {}
               self.unique_fft_properties = {}
               self.unique_cluster_properties = {}
               self.unique_shape_properties = {}
               self.unique_dynamic_properties = {}

               for track_id in self.filtered_track_ids:
                                    
                                    self._final_morphological_dynamic_vectors(track_id)
               self._compute_phenotypes()
               self._compute_track_vectors()

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
                                current_tracklets, current_tracklets_properties = self._tracklet_and_properties(all_dict_values, t, z, y, x, k, current_track_id, unique_id, current_tracklets, current_tracklets_properties)
                                    

                if str(track_id) in current_tracklets:
                        current_tracklets = np.asarray(current_tracklets[str(track_id)])
                        current_tracklets_properties = np.asarray(current_tracklets_properties[str(track_id)])
                        if len(current_tracklets.shape) == 2:
                            self.unique_tracks[track_id] = current_tracklets     
                            self.unique_track_properties[track_id] = current_tracklets_properties 

                
            
                        

        

                



               
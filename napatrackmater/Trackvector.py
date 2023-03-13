from . import TrackMate
from pathlib import Path 
import lxml.etree as et
import concurrent
import os
import numpy as np

class TrackVector(TrackMate):
       
        def __init__(self, master_xml_path: Path, t_minus: int = 0, t_plus: int = 10, x_start : int = 0, x_end: int = 10,
                    y_start: int = 0, y_end: int = 10, show_tracks: bool = True):
              
              self.master_xml_path = master_xml_path
              self.t_minus = t_minus
              self.t_plus = t_plus 
              self.x_start = x_start 
              self.x_end = x_end 
              self.y_start = y_start 
              self.y_end = y_end 
              self.show_tracks = show_tracks
              xml_parser = et.XMLParser(huge_tree=True)

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

            self._compute_cluster_phenotypes()                        
            self._temporal_plots_trackmate()   


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



                



               
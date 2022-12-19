import xml.etree.cElementTree as et
import codecs
import pandas as pd 
import numpy as np
from tifffile import imread
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from skimage.morphology import binary_dilation

class morphology_trajectory(object):
    
    def __init__(self, xml_path: str, spot_csv: str, track_csv: str, edges_csv: str, seg_image_path: str = None):
        
        self.xml_path = xml_path
        self.spot_csv = spot_csv 
        self.track_csv = track_csv 
        self.edges_csv = edges_csv
        self.seg_image_path = seg_image_path
        self.root = None 
        self.filtered_track_ids = None 
        self.tracks = None 
        self.settings = None
        self.spots = None 
        self.spot_dataset = None
        self.track_dataset = None
        self.edges_dataset = None 
        if seg_image_path is not None:
            self.seg_image = imread(self.seg_image_path).astype('uint16')
            self.ndim = len(self.seg_image.shape)
        else:
            self.seg_image = None
            self.ndim = None    
        
    def _extract_TM_info(self):
        
            self.root = et.fromstring(codecs.open(self.xml_path, 'r', 'utf8').read())
            self.filtered_track_ids = [int(track.get('TRACK_ID')) for track in self.root.find('Model').find('FilteredTracks').findall('TrackID')]
            #Extract the tracks from xml
            self.tracks = root.find('Model').find('AllTracks')
            self.settings = root.find('Settings').find('ImageData')
            #Extract the spots from xml
            self.spots = root.find('Model').find('AllSpots')

            #Information from the spots csv
            self.spot_dataset = pd.read_csv(self.spot_csv, delimiter = ',')[3:]
            self.spot_dataset_index = self.spot_dataset.index
            self.spot_dataset_keys = self.spot_dataset.keys()
            
            #Information from the tracks csv
            self.track_dataset = pd.read_csv(self.track_csv, delimiter = ',')[3:]
            self.track_dataset_index = self.track_dataset.index
            self.track_dataset_keys = track_dataset.keys()
            
            #Information from the edges csv
            self.edges_dataset = pd.read_csv(self.edges_csv, delimiter = ',')[3:]
            self.edges_dataset_index = self.edges_dataset.index
            self.edges_dataset_keys = self.edges_dataset.keys()
            
    def _get_neighbor_labels(self):
        
        assert self.seg_image is not None, f'A path to segmentation image is needed for obtaining neighbor label information, got: {self.seg_image_path} as path instead'       
        
        self.properties = regionprops(self.seg_image)
        self.Labels = [prop.label for prop in self.properties]
        
        self.neighbour_labels = {}
        
        for label_id in self.Labels:
            
            pixel_condition = (self.seg_image == label_id)
            indices = zip(*np.where(pixel_condition))
            _img = np.zeros_like(self.seg_image)
            for index in indices:
              _img[index] = 1
            _binary_image = find_boundaries(_img, mode = 'inner')
            _boundary_image = binary_dilation(_binary_image)
            _contact = (np.asarray(_boundary_image).astype(int) - np.asarray(_binary_image).astype(int))
            _indices = zip(*np.where(_contact > 0))
            contact_list = np.unique([self.seg_image[_index] for _index in _indices]).tolist()
            contact_list = list(filter(lambda num: num != 0 , contact_list))
            contact_list = list(filter(lambda num: num != label_id , contact_list))    
            self.neighbour_labels[label_id] = contact_list 
            
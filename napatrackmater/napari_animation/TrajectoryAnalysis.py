import xml.etree.cElementTree as et
import codecs
import pandas as pd 
import numpy as np
from tifffile import imread
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from skimage.morphology import binary_dilation

class morphology_trajectory(object):
	
	def __init__(self, xml_path: str, spot_csv: str, track_csv: str, edges_csv: str, seg_image_path: str = None, mask_image_path: str = None):
		
		self.xml_path = xml_path
		self.spot_csv = spot_csv 
		self.track_csv = track_csv 
		self.edges_csv = edges_csv
		self.seg_image_path = seg_image_path
		self.mask_image_path = mask_image_path
		self.root = None 
		self.filtered_track_ids = None 
		self.tracks = None 
		self.settings = None
		self.detector_settings = None
		self.spots = None 
		self.spot_dataset = None
		self.track_dataset = None
		self.edges_dataset = None 
  
		if self.seg_image_path is not None:
			self.seg_image = imread(self.seg_image_path).astype('uint16')
			self.ndim = len(self.seg_image.shape)
		else:
			self.seg_image = None
			self.ndim = None    
		if mask_image_path is not None:
			self.mask_image = imread(self.mask_image_path).astype('uint16')
		else:
			self.mask_image = None
		
		
	def create_interactive_analysis(self):
		
		self._extract_TM_info()
		self._get_neighbor_labels()    
		
	def _extract_TM_info(self):
		
			self.root = et.fromstring(codecs.open(self.xml_path, 'r', 'utf8').read())
			self.filtered_track_ids = [int(track.get('TRACK_ID')) for track in self.root.find('Model').find('FilteredTracks').findall('TrackID')]
			#Extract the tracks from xml
			self.tracks = root.find('Model').find('AllTracks')
			self.settings = root.find('Settings').find('ImageData')
			self.detector_settings = root.find('Settings').find('DetectorSettings')
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
			
			self.xcalibration = float(self.settings.get('pixelwidth'))
			self.ycalibration = float(self.settings.get('pixelheight'))
			self.zcalibration = float(self.settings('voxeldepth'))
			self.tcalibration = int(float(self.settings('timeinterval')))
			self.detection_channel = int(float(self.detector_settings('TARGET_CHANNEL')))
			self.detector = self.detector_settings('DETECTOR_NAME')
			assert self.detector == 'LABEL_IMAGE_DETECTOR', f'Only Label Image Detector is supported for this library, found: {self.detector} instead'
			
	def _neighbor_map(self, _current_image):
		
				self.properties = regionprops(_current_image)
				self.Labels = [prop.label for prop in self.properties]
				self.neighbour_map = {}
				
				for label_id in self.Labels:
					
					pixel_condition = (_current_image == label_id)
					indices = zip(*np.where(pixel_condition))
					_img = np.zeros_like(_current_image)
					for index in indices:
					   _img[index] = 1
					_binary_image = find_boundaries(_img, mode = 'inner')
					_boundary_image = binary_dilation(_binary_image)
					_contact = (np.asarray(_boundary_image).astype(int) - np.asarray(_binary_image).astype(int))
					_indices = zip(*np.where(_contact > 0))
					contact_list = np.unique([_current_image[_index] for _index in _indices]).tolist()
					contact_list = list(filter(lambda num: num != 0 , contact_list))
					contact_list = list(filter(lambda num: num != label_id , contact_list))    
					neighbour_map[label_id] = contact_list
				return neighbour_map
	
	def _get_neighbor_labels(self):
		
		assert self.seg_image is not None, f'A path to segmentation image is needed for obtaining neighbor label information, got: {self.seg_image_path} as path instead'       
		assert self.ndim > 2, f'For track analysis we require a 3D/2D + time image, got: {self.ndim} dimensional image instead'	
		self.temporal_neighbor_map = {}
		
		#Loop over time
		for t in range(self.seg_image.shape[0]):
				if self.ndim == 3:
					_current_image = self.seg_image[t,:,:]
				if self.ndim > 3:
					_current_image = self.seg_image[t,:,:,:]
				self.neighbour_map = self._neighbor_map(_current_image) 
				self.temporal_neighbor_map[t] =  self.neighbour_map
	  
			
	def _add_neighbor_spot_attribute(self):
	 
				current_spot_ids = self.spot_dataset['ID']
				current_spot_xs = (self.spot_dataset['POSITION_X'].astype('float')/self.xcalibration).astype('int')
				current_spot_ys = (self.spot_dataset['POSITION_Y'].astype('float')/self.ycalibration).astype('int')
				if self.ndim > 3:
	 				current_spot_zs = (self.spot_dataset['POSITION_Z'].astype('float')/self.zcalibration).astype('int')   
				current_spot_ts = (self.spot_dataset['FRAME'].astype('float')).astype('int') 
				contact_cell_list = []
				for i in range(len(current_spot_ids)):
					
					current_id = current_spot_ids[i]
					current_frame = current_spot_ts[i]
					current_location = (current_spot_zs[i], current_spot_ys[i], current_spot_xs[i])
					if self.ndim > 3:
	 					current_label = self.seg_image[current_frame,current_spot_zs[i], current_spot_ys[i], current_spot_xs[i]]
					if self.ndim == 3:
						current_label = self.seg_image[current_frame,current_spot_ys[i], current_spot_xs[i]]
					neighbor_map = self.temporal_neighbor_map[current_frame]
					contact_cell_labels = neighbour_map[current_label]
					#Create a list of neighbour labels for each spot in track  
					contact_cell_list.append(contact_cell_labels)   
		
						  
				self.spot_dataset['NEIGHBOR_SEG_LABELS'] =  contact_cell_list
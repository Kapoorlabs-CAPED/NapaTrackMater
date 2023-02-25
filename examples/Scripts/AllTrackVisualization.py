#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys

import numpy as np
import os
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import napari
import napatrackmater.bTrackmate as TM
import qtpy

# In[4]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/for_oneat_test/oneat_mari_principle_a60_mitosis_only.xml'
#Path to Segmentation image for extracting any track information from labels 
LabelImage = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/for_oneat_test/seg/for_tracking_tiltcorrected_cropped-1.tif'

RawImage = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/for_oneat_test/raw/for_tracking_tiltcorrected_cropped-1.tif'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/for_oneat_test/save_tracks/'
MaskImage = None 


# In[5]:


all_track_properties, split_points_times, Mask, calibration = TM.import_TM_XML(xml_path, RawImage, LabelImage,MaskImage)

print('showing tracks')
TM.ShowAllTracks(RawImage, LabelImage, Mask,savedir, calibration,all_track_properties)


# In[ ]:





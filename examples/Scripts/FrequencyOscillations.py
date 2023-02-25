#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import os
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import napari
from tifffile import imread
import napatrackmater.bTrackmate as TM


# In[2]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = 'output/tracking.xml' 
#Path to Segmentation image for extracting any track information from labels 
LabelImage = 'data/Seg.tif'
#Path to Raw image to display the tracks on (optional) else set it to None
RawImage = 'data/Raw.tif'
savedir = 'output/'
MaskImage = None #'data/Mask.tif'
mode = 'intensity'


# In[3]:


all_track_properties, Mask, calibration = TM.import_TM_XML(xml_path, RawImage, LabelImage, MaskImage)


# # Intensity Oscillations of Dividing tracks

# In[4]:


TM.TrackMateLiveTracks(RawImage, LabelImage, Mask,savedir, calibration,all_track_properties, True, mode = mode)


# # Intensity Oscillations of Non Dividing tracks

# In[ ]:


TM.TrackMateLiveTracks(RawImage, LabelImage, Mask,savedir, calibration,all_track_properties, False, mode = mode)


# In[ ]:





# In[ ]:





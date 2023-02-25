#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
import numpy as np
import os
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import napari
import napatrackmater.bTrackmate as TM


# In[2]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = '/Users/aimachine/MariTests/spot_visualization_test/t1_20.xml' 
#Path to Segmentation image for extracting any track information from labels 
LabelImage = '/Users/aimachine/MariTests/spot_visualization_test/t1_20.tif'#Path to Raw image to display the tracks on (optional) else set it to None
RawImage = '/Users/aimachine/MariTests/spot_visualization_test/t1_20.tif'
savedir = '/Users/aimachine/MariTests/spot_visualization_test/Results/'
MaskImage = None
nbins = 10


# In[3]:


all_track_properties, Mask, calibration = TM.import_TM_XML(xml_path, RawImage, LabelImage,MaskImage)


# # Visualize All tracks

# In[4]:


TM.ShowAllTracksGauss(RawImage, LabelImage, Mask,savedir, calibration,all_track_properties, nbins = nbins)


# In[ ]:





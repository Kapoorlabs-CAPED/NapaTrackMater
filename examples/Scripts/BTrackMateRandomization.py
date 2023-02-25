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
xml_path = '/Users/aimachine/DrosTracking/Tracking_dros-002.xml'
#Path to Segmentation image for extracting any track information from labels 
LabelImage = '/Users/aimachine/DrosTracking/Vollwt12-1.tif'
#Path to Raw image to display the tracks on (optional) else set it to None
RawImage = '/Users/aimachine/DrosTracking/wt_mov12_SW30-1.tif'
savedir = '/Users/aimachine/DrosTracking/'
MaskImage = None #'/media/sancere/Newton_Volume_1/ClaudiaAnalysis/Track_Analysis/105E_Day6/Analysis/Mask/105E_Mask.tif'


# In[3]:


all_track_properties, Mask, calibration = TM.import_TM_XML(xml_path, RawImage, LabelImage, MaskImage)


# # Visualize Dividing tracks

# In[4]:


TM.TrackMateLiveTracksGauss(RawImage, LabelImage, Mask,savedir, calibration,all_track_properties, True)


# # Visualize Non Dividing tracks

# In[ ]:


TM.TrackMateLiveTracksGauss(RawImage, LabelImage, Mask,savedir, calibration, all_track_properties, False)


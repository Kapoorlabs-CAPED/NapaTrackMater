#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
sys.path.append('../')
import numpy as np
import os
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import napari
import napatrackmater.bTrackmate as TM


# In[11]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = '/media/sancere/Newton_Volume_1/ClaudiaAnalysis/Track_Analysis/105E_Day6/Analysis/Results/DUP_105E_RawImage.xml' 
#Path to Segmentation image for extracting any track information from labels 
LabelImage = '/media/sancere/Newton_Volume_1/ClaudiaAnalysis/Track_Analysis/105E_Day6/Analysis/Seg/105E_Day6_SmartSeedsMask.tif'
#Path to Raw image to display the tracks on (optional) else set it to None
RawImage = '/media/sancere/Newton_Volume_1/ClaudiaAnalysis/Track_Analysis/105E_Day6/Analysis/Raw/105E_RawImage.tif'
savedir = '/media/sancere/Newton_Volume_1/ClaudiaAnalysis/Track_Analysis/105E_Day6/Analysis/Results/'
MaskImage = None #'/media/sancere/Newton_Volume_1/ClaudiaAnalysis/Track_Analysis/105E_Day6/Analysis/Mask/105E_Mask.tif'


# In[12]:


all_track_properties, Mask, calibration = TM.import_TM_XML(xml_path, RawImage, LabelImage, MaskImage)


# # Visualize Dividing tracks

# In[ ]:


TM.TrackMateLiveTracks(RawImage, LabelImage, Mask,savedir, calibration,all_track_properties, True)


# # Visualize Non Dividing tracks

# In[ ]:


TM.TrackMateLiveTracks(RawImage, LabelImage, Mask,savedir, calibration, all_track_properties, False)


# In[ ]:





# In[ ]:





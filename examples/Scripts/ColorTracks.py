#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
sys.path.append('../../')
import numpy as np
import os
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import napari
import napatrackmater.bTrackmate as TM
from pathlib import Path
get_ipython().run_line_magic('gui', 'qt')


# In[2]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = '/Users/aimachine/DrosTracking/Tracking_dros-002.xml' 
#Path to Segmentation image for extracting any track information from labels 
LabelImage = '/Users/aimachine/DrosTracking/Vollwt12-1.tif'
#Trackmate writes a spots and tracks file as csv
spot_csv = '/Users/aimachine/DrosTracking/spots_dros-1.csv'
track_csv = '/Users/aimachine/DrosTracking/tracks_dros-1.csv'
savedir = '/Users/aimachine/DrosTracking/'
Path(savedir).mkdir(exist_ok=True)
scale = 255


# In[3]:


TM.import_TM_XML_Relabel(xml_path,LabelImage,spot_csv, track_csv, savedir, scale = scale)


# In[ ]:





# In[ ]:





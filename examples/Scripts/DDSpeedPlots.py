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
xml_path = '/E/mainscipy//MariTests/full_tracks/for_traking_full.xml' 
#Trackmate writes a spots and tracks file as csv
spot_csv = '/E/mainscipy//MariTests/full_tracks/spots.csv'
links_csv = '/E/mainscipy//MariTests/full_tracks/links.csv'
savedir = '/E/mainscipy//MariTests/full_tracks/'
Path(savedir).mkdir(exist_ok=True)


# In[3]:


TM.import_TM_XML_statplots(xml_path,spot_csv, links_csv, savedir)


# In[ ]:





# In[ ]:





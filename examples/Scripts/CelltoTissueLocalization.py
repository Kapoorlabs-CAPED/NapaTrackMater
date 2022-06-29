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
import seaborn as sns
import napatrackmater.bTrackmate as TM
from pathlib import Path
get_ipython().run_line_magic('gui', 'qt')

plt.ion()


# In[2]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = '/Users/aimachine/mainscipy/MariTests/full_tracks/for_traking_full.xml' 
window_size = 10
view_angles = [15,10]


# In[3]:



Gradients = TM.import_TM_XML_Localization(xml_path, window_size = window_size, angle_1 = view_angles[0], angle_2 = view_angles[1])


# In[4]:


sns.histplot(Gradients, kde = True)


# In[ ]:





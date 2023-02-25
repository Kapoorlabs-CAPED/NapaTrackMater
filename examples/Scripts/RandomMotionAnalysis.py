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
xml_path = '/Users/aimachine/CBIAS-Demo/DUP_105E_RawImage.xml'
nbins = 10


# In[3]:


TM.import_TM_XML_Randomization(xml_path, nbins = nbins)


# In[ ]:





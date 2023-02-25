#!/usr/bin/env python
# coding: utf-8

# # Checkpoint Maker for Fiji bTrackmate
# 
# Create csv file of cell attributes for bTM 
# 
# The Fiji csv file has the headers:
# 
# T X Y Z Label Perimeter Volume Intensity ExtentX ExtentY ExtentZ
# 
# These columns are read in by bTrackmate csv reader and it creates
# 
# TM SpotCollection for tracking
# 

# In[1]:


from pathlib import Path
import sys
sys.path.append('../../')
import napatrackmater
from napatrackmater import bTrackmate


# In[2]:


RawPath = Path('/Users/aimachine/DrosTracking/wt_mov12_SW30-1.tif')
SegPath = Path('/Users/aimachine/DrosTracking/Vollwt12-1.tif')
MaskPath = None #Path('/Users/aimachine/Track_Analysis_105E_Day6/Mask/105E_Mask.tif')
savedir = Path('/Users/aimachine/DrosTracking/')
savedir.mkdir(exist_ok = True)
Name = RawPath.stem


# In[3]:


bTrackmate.CreateTrackCheckpoint(str(RawPath), str(SegPath), None, Name, str(savedir))


# In[ ]:





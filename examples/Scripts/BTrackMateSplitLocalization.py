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
import os
from tifffile import imread, imwrite
import napatrackmater
from napatrackmater import bTrackmate
import glob
import numpy as np
from natsort import natsorted


# Combine the split timepoints into a single segmentation image

# In[2]:


SplitSegPath = Path('/Users/aimachine/ClaudiaTracks_September/SmartSeedsMask/')
SavePath = '/Users/aimachine/ClaudiaTracks_September/Segmentation/'
SaveName = '20210912_143G_N1TOM_day5_region3_each10min_zoom07_FINAL_GFP'

Raw_path = os.path.join(SplitSegPath, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw = natsorted(filesRaw)
fname = filesRaw[0]
test_image_dimensions = imread(fname)
print(len(filesRaw))
BigImage = np.zeros([len(filesRaw),test_image_dimensions.shape[0],test_image_dimensions.shape[1],test_image_dimensions.shape[2] ]) 
count = 0
for fname in filesRaw:
         image = imread(fname)
         print(os.path.basename(os.path.splitext(fname)[0]))
         BigImage[count,:] = image
         count = count + 1
        
imwrite(SavePath + SaveName + '.tif', BigImage.astype('float16'))        


# In[5]:


RawPath = Path('/Users/aimachine/ClaudiaTracks_September/20210912_143G_N1TOM_day5_region3_each10min_zoom07_FINAL_GFP.tif')
SegPath = Path('/Users/aimachine/ClaudiaTracks_September/Segmentation/20210912_143G_N1TOM_day5_region3_each10min_zoom07_FINAL_GFP.tif')
MaskPath = None #Path('/Users/aimachine/Track_Analysis_105E_Day6/Mask/105E_Mask.tif')
savedir = Path('/Users/aimachine/ClaudiaTracks_September/Results/')
savedir.mkdir(exist_ok = True)
Name = RawPath.stem


# In[6]:


bTrackmate.CreateTrackCheckpoint(str(RawPath), str(SegPath), None, Name, str(savedir))


# In[ ]:





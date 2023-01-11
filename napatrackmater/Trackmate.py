
from tqdm import tqdm
import numpy as np 
import codecs
import xml.etree.ElementTree as et
import pandas as pd


class TrackMate(object):
    
    def __init__(self, xml_path, spot_csv_path, track_csv_path, edges_csv_path, AttributeBoxname, TrackAttributeBoxname, TrackidBox, image = None, mask = None, scale = 255 * 255):
        
        
        self.xml_path = xml_path
        self.spot_csv_path = spot_csv_path
        self.track_csv_path = track_csv_path 
        self.edges_csv_path = edges_csv_path
        self.image = image 
        self.mask = mask 
        self.AttributeBoxname = AttributeBoxname
        self.TrackAttributeBoxname = TrackAttributeBoxname
        self.TrackidBox = TrackidBox
        self.spot_dataset, self.spot_dataset_index = get_csv_data(self.spot_csv_path)
        self.track_dataset, self.track_dataset_index = get_csv_data(self.track_csv_path)
        self.edges_dataset, self.edges_dataset_index = get_csv_data(self.edges_csv_path)
        self.scale = scale
       
                                                        
        self.track_analysis_spot_keys = dict(
                spot_id="ID",
                track_id="TRACK_ID",
                quality="QUALITY",
                posix="POSITION_X",
                posiy="POSITION_Y",
                posiz="POSITION_Z",
                posit="POSITION_T",
                frame="FRAME",
                radius="RADIUS",
                mean_intensity_ch1="MEAN_INTENSITY_CH1",
                total_intensity_ch1="TOTAL_INTENSITY_CH1",
                mean_intensity_ch2="MEAN_INTENSITY_CH2",
                total_intensity_ch2="TOTAL_INTENSITY_CH2",
            )
        self.track_analysis_edges_keys = dict(
                spot_source_id="SPOT_SOURCE_ID",
                spot_target_id="SPOT_TARGET_ID",
                directional_change_rate="DIRECTIONAL_CHANGE_RATE",
                speed="SPEED",
                displacement="DISPLACEMENT",
                edge_time="EDGE_TIME",
                edge_x_location="EDGE_X_LOCATION",
                edge_y_location="EDGE_Y_LOCATION",
                edge_z_location="EDGE_Z_LOCATION",
            )
        self.track_analysis_track_keys = dict(
                number_spots="NUMBER_SPOTS",
                number_gaps="NUMBER_GAPS",
                number_splits="NUMBER_SPLITS",
                number_merges="NUMBER_MERGES",
                track_duration="TRACK_DURATION",
                track_start="TRACK_START",
                track_stop="TRACK_STOP",
                track_displacement="TRACK_DISPLACEMENT",
                track_x_location="TRACK_X_LOCATION",
                track_y_location="TRACK_Y_LOCATION",
                track_z_location="TRACK_Z_LOCATION",
                track_mean_speed="TRACK_MEAN_SPEED",
                track_max_speed="TRACK_MAX_SPEED",
                track_min_speed="TRACK_MIN_SPEED",
                track_median_speed="TRACK_MEDIAN_SPEED",
                track_std_speed="TRACK_STD_SPEED",
                track_mean_quality="TRACK_MEAN_QUALITY",
                total_track_distance="TOTAL_DISTANCE_TRAVELED",
                max_track_distance="MAX_DISTANCE_TRAVELED",
                mean_straight_line_speed="MEAN_STRAIGHT_LINE_SPEED",
                linearity_forward_progression="LINEARITY_OF_FORWARD_PROGRESSION",
                mean_directional_change_rate="MEAN_DIRECTIONAL_CHANGE_RATE",
            )

        self.frameid_key = self.track_analysis_spot_keys["frame"]
        self.zposid_key = self.track_analysis_spot_keys["posiz"]
        self.yposid_key = self.track_analysis_spot_keys["posiy"]
        self.xposid_key = self.track_analysis_spot_keys["posix"]
        self.spotid_key = self.track_analysis_spot_keys["spot_id"]
        self.trackid_key = self.track_analysis_spot_keys["track_id"]
        self.radius_key = self.track_analysis_spot_keys["radius"]
        self.quality_key = self.track_analysis_spot_keys["quality"]


        self.mean_intensity_ch1_key = self.track_analysis_spot_keys["mean_intensity_ch1"]
        self.mean_intensity_ch2_key = self.track_analysis_spot_keys["mean_intensity_ch2"]
        self.total_intensity_ch1_key = self.track_analysis_spot_keys["total_intensity_ch1"]
        self.total_intensity_ch2_key = self.track_analysis_spot_keys["total_intensity_ch2"]

        self.spot_source_id_key = self.track_analysis_edges_keys["spot_source_id"]
        self.spot_target_id_key = self.track_analysis_edges_keys["spot_target_id"]
        self.directional_change_rate_key = self.track_analysis_edges_keys["directional_change_rate"] 
        self.speed_key = self.track_analysis_edges_keys["speed"],
        self.displacement_key = self.track_analysis_edges_keys["displacement"]
        self.edge_time_key = self.track_analysis_edges_keys["edge_time"]
        self.edge_x_location_key = self.track_analysis_edges_keys["edge_x_location"]
        self.edge_y_location_key = self.track_analysis_edges_keys["edge_y_location"]
        self.edge_z_location_key = self.track_analysis_edges_keys["edge_z_location"]
        
        self._get_xml_data()
        self._get_attributes()
        self._temporal_plots_trackmate()



    def _get_attributes(self):
            
             self.Attributeids, self.AllValues =  get_spot_dataset(self.spot_dataset, self.spot_dataset_index, self.track_analysis_spot_keys, self.xcalibration, self.ycalibration, self.zcalibration, self.AttributeBoxname)
        
             self.TrackAttributeids, self.AllTrackValues = get_track_dataset(self.track_dataset, self.track_dataset_index, self.track_analysis_spot_keys, self.track_analysis_track_keys, self.TrackAttributeBoxname, self.scale)
             
             self.AllEdgesValues = get_edges_dataset(self.edges_dataset, self.edges_dataset_index, self.track_analysis_spot_keys, self.track_analysis_edges_keys)
        
    def _get_xml_data(self):

                self.xml_content = et.fromstring(codecs.open(self.xml_path, "r", "utf8").read())

                self.Uniqueobjects = {}
                self.Uniqueproperties = {}
                self.AllTrackIds = []
                self.DividingTrackIds = []
                self.NormalTrackIds = []
                self.all_track_properties = []
                self.split_points_times = []
                self.filtered_track_ids = [
                    int(track.get(self.trackid_key))
                    for track in self.xml_content.find("Model")
                    .find("FilteredTracks")
                    .findall("TrackID")
                ]
                
                self.AllTrackIds.append("")
                self.DividingTrackIds.append("")
                self.NormalTrackIds.append("")
                
                self.AllTrackIds.append(self.TrackidBox)
                self.DividingTrackIds.append(self.TrackidBox)
                self.NormalTrackIds.append(self.TrackidBox)
                
                
                self.Spotobjects = self.xml_content.find('Model').find('AllSpots')
                # Extract the tracks from xml
                self.tracks = self.xml_content.find("Model").find("AllTracks")
                self.settings = self.xml_content.find("Settings").find("ImageData")
                self.xcalibration = float(self.settings.get("pixelwidth"))
                self.ycalibration = float(self.settings.get("pixelheight"))
                self.zcalibration = float(self.settings.get("voxeldepth"))
                self.tcalibration = int(float(self.settings.get("timeinterval")))

               

                if  self.mask is not None and self.image is not None:
                    if len(self.mask.shape) < len(self.image.shape):
                        # T Z Y X
                        self.update_mask = np.zeros(
                            [
                                self.image.shape[0],
                                self.image.shape[1],
                                self.image.shape[2],
                                self.image.shape[3],
                            ]
                        )
                        for i in range(0, self.update_mask.shape[0]):
                            for j in range(0, self.update_mask.shape[1]):

                                self.update_mask[i, j, :, :] = self.mask[i, :, :]
                                
                    else:
                        self.update_mask = self.mask
                        
                    self.mask = self.update_mask.astype('uint16')

                    self.timed_mask, self.boundary = boundary_points(self.mask, self.xcalibration, self.ycalibration, self.zcalibration)
                else:
                    self.timed_mask = None
                    self.boundary = None    
                        
            

                for frame in self.Spotobjects.findall('SpotsInFrame'):

                    for Spotobject in frame.findall('Spot'):
                        # Create object with unique cell ID
                        cell_id = int(Spotobject.get(self.spotid_key))
                        # Get the TZYX location of the cells in that frame
                        
                        self.Uniqueobjects[cell_id] = [
                            Spotobject.get(self.frameid_key),
                            Spotobject.get(self.zposid_key),
                            Spotobject.get(self.yposid_key),
                            Spotobject.get(self.xposid_key),
                        ]
                        
                        # Get other properties associated with the Spotobject
                        TOTAL_INTENSITY_CH1 = Spotobject.get(self.total_intensity_ch1_key)
                        MEAN_INTENSITY_CH1 = Spotobject.get(self.mean_intensity_ch1_key)
                        Radius = Spotobject.get(self.radius_key)
                        QUALITY = Spotobject.get(self.quality_key)
                                
                        self.Uniqueproperties[cell_id] = [
                            
                            TOTAL_INTENSITY_CH1,
                            MEAN_INTENSITY_CH1,
                            Spotobject.get(self.frameid_key),
                            Radius,
                            QUALITY
                        ]

                
                
                for track in self.tracks.findall('Track'):

                    track_id = int(track.get(self.trackid_key))

                    self.spot_object_source_target = []
                    if track_id in self.filtered_track_ids:
                        
                        
                        for edge in track.findall('Edge'):

                            source_id = edge.get(self.spot_source_id_key)
                            target_id = edge.get(self.spot_target_id_key)
                            if int(source_id) in self.Uniqueproperties:
                                TOTAL_INTENSITY_CH1, MEAN_INTENSITY_CH1,Position_T,Radius,QUALITY = self.Uniqueproperties[int(source_id)]
                            else:
                                TOTAL_INTENSITY_CH1, MEAN_INTENSITY_CH1,Position_T,Radius,QUALITY = self.Uniqueproperties[int(target_id)] 
                            edge_time = Position_T
                            directional_rate_change = edge.get(self.directional_change_rate_key)
                            speed = edge.get(self.speed_key)

                            self.spot_object_source_target.append(
                                [source_id, target_id, edge_time, directional_rate_change, speed]
                            )

                        # Sort the tracks by edge time
                        self.spot_object_source_target = sorted(
                            self.spot_object_source_target, key=sortTracks, reverse=False
                        )
                        # Get all the IDs, uniquesource, targets attached, leaf, root, splitpoint IDs
                        split_points, split_times, root_leaf = Multiplicity(self.spot_object_source_target)

                        # Determine if a track has divisions or none
                        if len(split_points) > 0:
                            split_points = split_points[::-1]
                            DividingTrajectory = True
                        else:
                            DividingTrajectory = False
                    
                        if DividingTrajectory == True:
                            self.AllTrackIds.append(str(track_id))
                            self.DividingTrackIds.append(str(track_id))
                            self.tracklets = analyze_dividing_tracklets(
                                root_leaf, split_points, self.spot_object_source_target
                            )
                            for i in range(len(split_points)):
                                self.split_points_times.append([split_points[i], split_times[i]])
                        if DividingTrajectory == False:
                            self.AllTrackIds.append(str(track_id))
                            self.NormalTrackIds.append(str(track_id))
                            self.tracklets = analyze_non_dividing_tracklets(
                                root_leaf, self.spot_object_source_target
                            )

                        # for each tracklet get real_time,z,y,x,total_intensity, mean_intensity, cellradius, distance, prob_inside
                        self.location_prop_dist = tracklet_properties(
                            self.tracklets,
                            self.Uniqueobjects,
                            self.Uniqueproperties,
                            self.mask,
                            None,
                            self.timed_mask,
                            [self.xcalibration, self.ycalibration, self.zcalibration, self.tcalibration],
                            DividingTrajectory
                        )
                        self.all_track_properties.append([track_id, self.location_prop_dist, DividingTrajectory])
                        
                        
                
    def _temporal_plots_trackmate(self):
    
    
    
                self.Attr = {}
                sourceid_key = self.track_analysis_edges_keys["spot_source_id"]
                dcr_key = self.track_analysis_edges_keys["directional_change_rate"]
                speed_key = self.track_analysis_edges_keys["speed"]
                disp_key = self.track_analysis_edges_keys["displacement"]

                starttime = int(min(self.AllValues[self.frameid_key]))
                endtime = int(max(self.AllValues[self.frameid_key]))

                for (
                    sourceid,
                    dcrid,
                    speedid,
                    dispid,
                    zposid,
                    yposid,
                    xposid,
                    radiusid,
                    meanintensitych1id,
                    meanintensitych2id,
                ) in zip(
                    self.AllEdgesValues[sourceid_key],
                    self.AllEdgesValues[dcr_key],
                    self.AllEdgesValues[speed_key],
                    self.AllEdgesValues[disp_key],
                    self.AllValues[self.zposid_key],
                    self.AllValues[self.yposid_key],
                    self.AllValues[self.xposid_key],
                    self.AllValues[self.radius_key],
                    self.AllValues[self.mean_intensity_ch1_key],
                    self.AllValues[self.mean_intensity_ch2_key],
                ):

                    self.Attr[int(sourceid)] = [
                        dcrid,
                        speedid,
                        dispid,
                        zposid,
                        yposid,
                        xposid,
                        radiusid,
                        meanintensitych1id,
                        meanintensitych2id,
                    ]


                self.Time = []
                self.Alldcrmean = []
                self.Allspeedmean = []
                self.Allradiusmean = []
                self.AllCurmeaninch1mean = []
                self.AllCurmeaninch2mean = []
                self.Alldispmeanpos = []
                self.Alldispmeanneg = []
                self.Alldispmeanposx = []
                self.Alldispmeanposy = []
                self.Alldispmeannegx = []
                self.Alldispmeannegy = []
                self.Alldcrvar = []
                self.Allspeedvar = []
                self.Allradiusvar = []
                self.AllCurmeaninch1var = []
                self.AllCurmeaninch2var = []
                self.Alldispvarpos = []
                self.Alldispvarneg = []
                self.Alldispvarposy = []
                self.Alldispvarnegy = []
                self.Alldispvarposx = []
                self.Alldispvarnegx = []

                for i in tqdm(range(starttime, endtime), total=endtime - starttime):

                    Curdcr = []
                    Curspeed = []
                    Curdisp = []
                    Curdispz = []
                    Curdispy = []
                    Curdispx = []
                    Currpos = []
                    Curmeaninch1 = []
                    Curmeaninch2 = []
                    for spotid, trackid, frameid in zip(
                        self.AllValues[self.spotid_key],
                        self.AllValues[self.trackid_key],
                        self.AllValues[self.frameid_key],
                    ):

                        if i == int(frameid):
                            if int(spotid) in self.Attr:
                                (
                                    dcr,
                                    speed,
                                    disp,
                                    zpos,
                                    ypos,
                                    xpos,
                                    rpos,
                                    meaninch1pos,
                                    meaninch2pos,
                                ) = self.Attr[int(spotid)]
                                if dcr is not None:
                                    Curdcr.append(dcr)

                                if speed is not None:
                                    Curspeed.append(speed)
                                if disp is not None:
                                    Curdisp.append(disp)
                                if zpos is not None:
                                    Curdispz.append(zpos)
                                if ypos is not None:
                                    Curdispy.append(ypos)

                                if xpos is not None:
                                    Curdispx.append(xpos)
                                if rpos is not None:
                                    Currpos.append(rpos)
                                if meaninch1pos is not None:
                                    Curmeaninch1.append(meaninch1pos)

                                if meaninch2pos is not None:
                                    Curmeaninch2.append(meaninch2pos)

                    dispZ = np.abs(np.diff(Curdispz))
                    dispY = np.abs(np.diff(Curdispy))
                    dispX = np.abs(np.diff(Curdispx))
                    
                    meanCurdcr = np.mean(Curdcr)
                    varCurdcr = np.std(Curdcr)
                    meanCurspeed = np.mean(Curspeed)
                    varCurspeed = np.std(Curspeed)
                    meanCurrpos = np.mean(Currpos)
                    varCurrpos = np.std(Currpos)
                    meanCurmeaninch1 = np.mean(Curmeaninch1)
                    varCurmeaninch1 = np.std(Curmeaninch1)
                    meanCurmeaninch2 = np.mean(Curmeaninch2)
                    varCurmeaninch2 = np.std(Curmeaninch2)
                    meanCurdisp = np.mean(dispZ)
                    varCurdisp = np.std(dispZ)
                    meanCurdispy = np.mean(dispY)
                    varCurdispy = np.std(dispY)
                    meanCurdispx = np.mean(dispX)
                    varCurdispx = np.std(dispX)
                    
                    self.Time.append(i * self.tcalibration)
                    self.Alldcrmean.append(meanCurdcr)
                    self.Alldcrvar.append(varCurdcr)
                    self.Allspeedmean.append(meanCurspeed)
                    self.Allspeedvar.append(varCurspeed)
                    self.Allradiusmean.append(meanCurrpos)
                    self.Allradiusvar.append(varCurrpos)
                    self.AllCurmeaninch1mean.append(meanCurmeaninch1)
                    self.AllCurmeaninch1var.append(varCurmeaninch1)
                    self.AllCurmeaninch2mean.append(meanCurmeaninch2)
                    self.AllCurmeaninch2var.append(varCurmeaninch2)

                    

                    self.Alldispmeanpos.append(meanCurdisp)
                    self.Alldispvarpos.append(varCurdisp)
                    self.Alldispmeanposy.append(meanCurdispy)
                    self.Alldispvarposy.append(varCurdispy)
                    self.Alldispmeanposx.append(meanCurdispx)
                    self.Alldispvarposx.append(varCurdispx)
                   
                            
                            
              

def analyze_non_dividing_tracklets(root_leaf, spot_object_source_target):

    non_dividing_tracklets = []
    if len(root_leaf) > 0:
        Root = root_leaf[0]
        Leaf = root_leaf[-1]
        tracklet = []
        trackletspeed = []
        trackletdirection = []
        trackletid = 0
        # For non dividing trajectories iterate from Root to the only Leaf
        while Root != Leaf:
            for (
                source_id,
                target_id,
                edge_time,
                directional_rate_change,
                speed,
            ) in spot_object_source_target:
                if Root == source_id:
                    tracklet.append(source_id)
                    trackletspeed.append(speed)
                    trackletdirection.append(directional_rate_change)
                    Root = target_id
                    if Root == Leaf:
                        break
                else:
                    break
                
        non_dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
    return non_dividing_tracklets


def analyze_dividing_tracklets(root_leaf, split_points, spot_object_source_target):

    dividing_tracklets = []
    # Make tracklets
    Root = root_leaf[0]

    visited = []
    # For the root we need to go forward
    tracklet = []
    trackletspeed = []
    trackletdirection = []
    tracklet.append(Root)
    trackletspeed.append(0)
    trackletdirection.append(0)
    trackletid = 1
    RootCopy = Root
    visited.append(Root)
    while RootCopy not in split_points and RootCopy not in root_leaf[1:]:
            for (
                    source_id,
                    target_id,
                    edge_time,
                    directional_rate_change,
                    speed,
                ) in spot_object_source_target:
                    # Search for the target id corresponding to leaf
                    if RootCopy == source_id:

                        # Once we find the leaf we move a step fwd to its target to find its target
                        RootCopy = target_id
                        if RootCopy in split_points:
                            break
                        if RootCopy in visited:
                            break
                        visited.append(target_id)
                        tracklet.append(source_id)
                        trackletspeed.append(speed)
                        trackletdirection.append(directional_rate_change)
    dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
    trackletid = 2

    # Exclude the split point near root
    for i in range(0, len(split_points )):
        Start = split_points[i]
        tracklet = []
        trackletspeed = []
        trackletdirection = []
        tracklet.append(Start)
        trackletspeed.append(0)
        trackletdirection.append(0)
        Othersplit_points = split_points.copy()
        Othersplit_points.pop(i)
        
        for (
                        source_id,
                        target_id,
                        edge_time,
                        directional_rate_change,
                        speed,
                    ) in spot_object_source_target:
                        
                        if Start in visited:
                           break 
                        if Start in Othersplit_points:
                            break
                        if Start == target_id:

                            Start = source_id
                            
                            tracklet.append(source_id)
                            visited.append(source_id)
                            trackletspeed.append(speed)
                            trackletdirection.append(directional_rate_change)
                            

        dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
        trackletid = trackletid + 1

    for i in range(0, len(root_leaf)):
        leaf = root_leaf[i]
        # For leaf we need to go backward
        tracklet = []
        trackletspeed = []
        trackletdirection = []
        tracklet.append(leaf)
        trackletspeed.append(0)
        trackletdirection.append(0)
        while leaf not in split_points and leaf != Root:
            for (
                    source_id,
                    target_id,
                    edge_time,
                    directional_rate_change,
                    speed,
                ) in spot_object_source_target:
                    # Search for the target id corresponding to leaf
                    if leaf == target_id:
                        # Include the split points here

                        # Once we find the leaf we move a step back to its source to find its source
                        leaf = source_id
                        if leaf in split_points:
                            break
                        if leaf in visited:
                            break
                        if leaf == Root:
                            break
                        visited.append(source_id)
                        tracklet.append(source_id)
                        trackletspeed.append(speed)
                        trackletdirection.append(directional_rate_change)
        dividing_tracklets.append([trackletid, tracklet, trackletspeed, trackletdirection])
        trackletid = trackletid + 1

    return dividing_tracklets


def tracklet_properties(
    alltracklets,
    Uniqueobjects,
    Uniqueproperties,
    Mask,
    seg_image,
    TimedMask,
    calibration,
    DividingTrajectory
):

    location_prop_dist = {}

    for i in range(0, len(alltracklets)):
        current_location_prop_dist = []
        trackletid, tracklets, trackletspeeds, trackletdirections = alltracklets[i]
        location_prop_dist[trackletid] = [trackletid]
        for k in range(0, len(tracklets)):

            tracklet = tracklets[k]
            trackletspeed = trackletspeeds[k]
            trackletdirection = trackletdirections[k]
            cell_source_id = tracklet
            frame, z, y, x = Uniqueobjects[int(cell_source_id)]
         
            total_intensity, mean_intensity, real_time, cellradius, pixel_volume = Uniqueproperties[
                int(cell_source_id)
            ]

            if Mask is not None:

                if len(Mask.shape) == 4:  
                  testlocation = (z, y, x)
                if len(Mask.shape) == 3:  
                  testlocation = (y, x)

                tree, indices, masklabel, masklabelvolume = TimedMask[str(int(float(frame)/ calibration[3]))]
                if len(Mask.shape) == 4:
                        region_label = Mask[
                            int(float(frame) / calibration[3]),
                            int(float(z) / calibration[2]),
                            int(float(y) / calibration[1]),
                            int(float(x) / calibration[0]),
                        ]
                if len(Mask.shape) == 3:
                        region_label = Mask[
                            int(float(frame) / calibration[3]),
                            int(float(y) / calibration[1]),
                            int(float(x) / calibration[0]),
                        ]
                for k in range(0, len(masklabel)):
                    currentlabel = masklabel[k]
                    currentvolume = masklabelvolume[k]
                    currenttree = tree[k]
                    # Get the location and distance to the nearest boundary point
                    distance, location = currenttree.query(testlocation)
                    distance = max(0, distance - float(cellradius))
                    if currentlabel == region_label and region_label > 0:
                        prob_inside = prob_sigmoid(distance)
                    else:

                        prob_inside = 0
            else:
                distance = 0
                prob_inside = 0

            current_location_prop_dist.append(
                [
                    frame,
                    z,
                    y,
                    x,
                    total_intensity,
                    mean_intensity,
                    cellradius,
                    distance,
                    prob_inside,
                    trackletspeed,
                    trackletdirection,
                    DividingTrajectory
                ]
            )

        location_prop_dist[trackletid].append(current_location_prop_dist)

    return location_prop_dist


          
        
        
        
def boundary_points(mask, xcalibration, ycalibration, zcalibration):

    ndim = len(mask.shape)

    # YX shaped object
    if ndim == 2:
        mask = label(mask)
        labels = []
        size = []
        tree = []
        properties = measure.regionprops(mask, mask)
        for prop in properties:

            labelimage = prop.image
            regionlabel = prop.label
            sizey = abs(prop.bbox[0] - prop.bbox[2]) * xcalibration
            sizex = abs(prop.bbox[1] - prop.bbox[3]) * ycalibration
            volume = sizey * sizex
            radius = math.sqrt(volume / math.pi)
            boundary = find_boundaries(labelimage)
            indices = np.where(boundary > 0)
            real_indices = np.transpose(np.asarray(indices)).copy()
            for j in range(0, len(real_indices)):

                real_indices[j][0] = real_indices[j][0] * xcalibration
                real_indices[j][1] = real_indices[j][1] * ycalibration

            tree.append(spatial.cKDTree(real_indices))

            if regionlabel not in labels:
                labels.append(regionlabel)
                size.append(radius)
        # This object contains list of all the points for all the labels in the Mask image with the label id and volume of each label
        timed_mask[str(0)] = [tree, indices, labels, size]

    # TYX shaped object
    if ndim == 3:

        Boundary = find_boundaries(mask)
        for i in tqdm(range(0, mask.shape[0])):

            mask[i, :] = label(mask[i, :])
            properties = measure.regionprops(mask[i, :], mask[i, :])
            labels = []
            size = []
            tree = []
            for prop in properties:

                labelimage = prop.image
                regionlabel = prop.label
                sizey = abs(prop.bbox[0] - prop.bbox[2]) * ycalibration
                sizex = abs(prop.bbox[1] - prop.bbox[3]) * xcalibration
                volume = sizey * sizex
                radius = math.sqrt(volume / math.pi)
                boundary = find_boundaries(labelimage)
                indices = np.where(boundary > 0)
                real_indices = np.transpose(np.asarray(indices)).copy()
                for j in range(0, len(real_indices)):

                    real_indices[j][0] = real_indices[j][0] * ycalibration
                    real_indices[j][1] = real_indices[j][1] * xcalibration

                tree.append(spatial.cKDTree(real_indices))
                if regionlabel not in labels:
                    labels.append(regionlabel)
                    size.append(radius)

            timed_mask[str(i)] = [tree, indices, labels, size]
            
    # TZYX shaped object
    if ndim == 4:

        Boundary = np.zeros(
            [mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]]
        )
        results = []
        x_ls = range(0, mask.shape[0])
        
        results.append(parallel_map(mask, xcalibration, ycalibration, zcalibration, Boundary, i) for i in tqdm(x_ls))
        da.delayed(results).compute()
        

    return timed_mask, Boundary        


           
def Multiplicity(spot_object_source_target):

    split_points = []
    split_times = []
    root_leaf = []
    sources = []
    targets = []
    scount = 0

    for (
        source_id,
        target_id,
        sourcetime,
        directional_rate_change,
        speed,
    ) in spot_object_source_target:

        sources.append(source_id)
        targets.append(target_id)

    root_leaf.append(sources[0])
    for (
        source_id,
        target_id,
        sourcetime,
        directional_rate_change,
        speed,
    ) in spot_object_source_target:

        if target_id not in sources:

            root_leaf.append(target_id)

        for (
            sec_source_id,
            sec_target_id,
            sec_sourcetime,
            sec_directional_rate_change,
            sec_speed,
        ) in spot_object_source_target:
            if source_id == sec_source_id:
                scount = scount + 1
        if scount > 1:
            split_points.append(source_id)
            split_times.append(sourcetime)
        scount = 0

    return split_points, split_times, root_leaf



def sortTracks(List):

    return int(float(List[2]))


def get_csv_data(csv):

        dataset = pd.read_csv(
            csv, delimiter=",", encoding="unicode_escape", low_memory=False
        )[3:]
        dataset_index = dataset.index
        return dataset, dataset_index
    
def get_spot_dataset(spot_dataset, spot_dataset_index, track_analysis_spot_keys, xcalibration, ycalibration, zcalibration, AttributeBoxname):

        AllValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        posix = track_analysis_spot_keys["posix"]
        posiy = track_analysis_spot_keys["posiy"]
        posiz = track_analysis_spot_keys["posiz"]
        frame = track_analysis_spot_keys["frame"]
        Tid = spot_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = spot_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1

        AllValues[track_id] = Tid
        LocationX = (
            spot_dataset[posix].astype("float") / xcalibration
        ).astype("int")
        LocationY = (
            spot_dataset[posiy].astype("float") / ycalibration
        ).astype("int")
        LocationZ = (
            spot_dataset[posiz].astype("float") / zcalibration
        ).astype("int")
        LocationT = (spot_dataset[frame].astype("float")).astype("int")
        AllValues[posix] = LocationX
        AllValues[posiy] = LocationY
        AllValues[posiz] = LocationZ
        AllValues[frame] = LocationT

        for k in track_analysis_spot_keys.values():

            if (
                k != track_id
                and k != posix
                and k != posiy
                and k != posiz
                and k != frame
            ):

                AllValues[k] = spot_dataset[k].astype("float")

        Attributeids = []
        Attributeids.append(AttributeBoxname)
        for attributename in track_analysis_spot_keys.keys():
            Attributeids.append(attributename)    
            
        
        return Attributeids, AllValues     
    
def get_track_dataset(track_dataset, track_dataset_index, track_analysis_spot_keys, track_analysis_track_keys, TrackAttributeBoxname, scale):

        AllTrackValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        Tid = track_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = track_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1
        AllTrackValues[track_id] = Tid
        for (k, v) in track_analysis_track_keys.items():

            if k != track_id:
                x = track_dataset[v].astype("float")
                minval = min(x)
                maxval = max(x)

                if minval > 0 and maxval <= 1:

                    x = normalizeZeroOne(x, scale=scale)

                AllTrackValues[k] = x

        TrackAttributeids = []
        TrackAttributeids.append(TrackAttributeBoxname)
        for attributename in track_analysis_track_keys.keys():
            TrackAttributeids.append(attributename)    
    
        return TrackAttributeids, AllTrackValues
    
def get_edges_dataset(edges_dataset, edges_dataset_index, track_analysis_spot_keys, track_analysis_edges_keys):

        AllEdgesValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        Tid = edges_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = edges_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1
        AllEdgesValues[track_id] = Tid

        for k in track_analysis_edges_keys.values():

            if k != track_id:
                x = edges_dataset[k].astype("float")

                AllEdgesValues[k] = x   
         
        return AllEdgesValues   
    
def normalizeZeroOne(x, scale = 255 * 255):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x * scale          
    
    
def all_tracks(TrackLayerTracklets, trackid, alltracklets, xcalibration, ycalibration, zcalibration, tcalibration):

        TrackLayerTracklets[trackid] = [trackid]
        list_tracklets = []
        for (trackletid, tracklets) in alltracklets.items():

            Locationtracklets = tracklets[1]
            if len(Locationtracklets) > 0:
                Locationtracklets = sorted(
                    Locationtracklets, key=sortFirst, reverse=False
                )
                for tracklet in Locationtracklets:
                    (
                        t,
                        z,
                        y,
                        x,
                        total_intensity,
                        mean_intensity,
                        cellradius,
                        distance,
                        prob_inside,
                        trackletspeed,
                        trackletdirection,
                        DividingTrajectory,
                    ) = tracklet
                   
                    list_tracklets.append(
                            [
                                trackletid, 
                                int(float(t)) ,
                                int(float(z)/zcalibration),
                                int(float(y)/ycalibration),
                                int(float(x)/xcalibration),
                            ]
                        )
        list_tracklets = sorted(
                    list_tracklets, key=sortFirst, reverse=False
                )            
        TrackLayerTracklets[trackid].append(list_tracklets)

        return TrackLayerTracklets    
    
def sortFirst(List):

    return int(float(List[0]))    
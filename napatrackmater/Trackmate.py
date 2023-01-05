
from tqdm import tqdm
import numpy as np 

track_analysis_spot_keys = dict(
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
track_analysis_edges_keys = dict(
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
track_analysis_track_keys = dict(
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

def temporal_plots_trackmate(AllValues, AllEdgesValues, tcalibration):
    
    
    
        Attr = {}

        frameid_key = track_analysis_spot_keys["frame"]
        zposid_key = track_analysis_spot_keys["posiz"]
        yposid_key = track_analysis_spot_keys["posiy"]
        xposid_key = track_analysis_spot_keys["posix"]
        spotid_key = track_analysis_spot_keys["spot_id"]
        trackid_key = track_analysis_spot_keys["track_id"]
        radius_key = track_analysis_spot_keys["radius"]
        mean_intensity_ch1 = track_analysis_spot_keys["mean_intensity_ch1"]
        mean_intensity_ch2 = track_analysis_spot_keys["mean_intensity_ch2"]

        sourceid_key = track_analysis_edges_keys["spot_source_id"]
        dcr_key = track_analysis_edges_keys["directional_change_rate"]
        speed_key = track_analysis_edges_keys["speed"]
        disp_key = track_analysis_edges_keys["displacement"]

        starttime = int(min(AllValues[frameid_key]))
        endtime = int(max(AllValues[frameid_key]))

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
            AllEdgesValues[sourceid_key],
            AllEdgesValues[dcr_key],
            AllEdgesValues[speed_key],
            AllEdgesValues[disp_key],
            AllValues[zposid_key],
            AllValues[yposid_key],
            AllValues[xposid_key],
            AllValues[radius_key],
            AllValues[mean_intensity_ch1],
            AllValues[mean_intensity_ch2],
        ):

            Attr[int(sourceid)] = [
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

        Timedcr = []
        Timespeed = []
        Timeradius = []
        TimeCurmeaninch2 = []
        TimeCurmeaninch1 = []
        Timedisppos = []
        Timedispneg = []

        Timedispposy = []
        Timedispnegy = []

        Timedispposx = []
        Timedispnegx = []

        Alldcrmean = []
        Allspeedmean = []
        Allradiusmean = []
        AllCurmeaninch1mean = []
        AllCurmeaninch2mean = []
        Alldispmeanpos = []
        Alldispmeanneg = []

        Alldispmeanposx = []
        Alldispmeanposy = []

        Alldispmeannegx = []
        Alldispmeannegy = []

        Alldcrvar = []
        Allspeedvar = []
        Allradiusvar = []
        AllCurmeaninch1var = []
        AllCurmeaninch2var = []
        Alldispvarpos = []
        Alldispvarneg = []

        Alldispvarposy = []
        Alldispvarnegy = []

        Alldispvarposx = []
        Alldispvarnegx = []

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
                AllValues[spotid_key],
                AllValues[trackid_key],
                AllValues[frameid_key],
            ):

                if i == int(frameid):
                    if int(spotid) in Attr:
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
                        ) = Attr[int(spotid)]
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
            if meanCurdcr is not None:
                Alldcrmean.append(meanCurdcr)
                Alldcrvar.append(varCurdcr)
                Timedcr.append(i * tcalibration)

            meanCurspeed = np.mean(Curspeed)
            varCurspeed = np.std(Curspeed)
            if meanCurspeed is not None:

                Allspeedmean.append(meanCurspeed)
                Allspeedvar.append(varCurspeed)
                Timespeed.append(i * tcalibration)

            meanCurrpos = np.mean(Currpos)
            varCurrpos = np.std(Currpos)
            
            if meanCurrpos is not None:

                Allradiusmean.append(meanCurrpos)
                Allradiusvar.append(varCurrpos)
                Timeradius.append(i * tcalibration)

            meanCurmeaninch1 = np.mean(Curmeaninch1)
            varCurmeaninch1 = np.std(Curmeaninch1)
            if meanCurmeaninch1 is not None:

                AllCurmeaninch1mean.append(meanCurmeaninch1)
                AllCurmeaninch1var.append(varCurmeaninch1)
                TimeCurmeaninch1.append(i * tcalibration)

            meanCurmeaninch2 = np.mean(Curmeaninch2)
            varCurmeaninch2 = np.std(Curmeaninch2)
            if meanCurmeaninch2 is not None:

                AllCurmeaninch2mean.append(meanCurmeaninch2)
                AllCurmeaninch2var.append(varCurmeaninch2)
                TimeCurmeaninch2.append(i * tcalibration)

            meanCurdisp = np.mean(dispZ)
            varCurdisp = np.std(dispZ)

            meanCurdispy = np.mean(dispY)
            varCurdispy = np.std(dispY)

            meanCurdispx = np.mean(dispX)
            varCurdispx = np.std(dispX)

            if meanCurdisp is not None:
                if meanCurdisp >= 0:
                    Alldispmeanpos.append(meanCurdisp)
                    Alldispvarpos.append(varCurdisp)
                    Timedisppos.append(i * tcalibration)
                elif meanCurdisp < 0:
                    Alldispmeanneg.append(meanCurdisp)
                    Alldispvarneg.append(varCurdisp)
                    Timedispneg.append(i * tcalibration)

            if meanCurdispy is not None:
                if meanCurdispy >= 0:
                    Alldispmeanposy.append(meanCurdispy)
                    Alldispvarposy.append(varCurdispy)
                    Timedispposy.append(i * tcalibration)
                elif meanCurdispy < 0:
                    Alldispmeannegy.append(meanCurdispy)
                    Alldispvarnegy.append(varCurdispy)
                    Timedispnegy.append(i * tcalibration)

            if meanCurdispx is not None:
                if meanCurdispx >= 0:
                    Alldispmeanposx.append(meanCurdispx)
                    Alldispvarposx.append(varCurdispx)
                    Timedispposx.append(i * tcalibration)
                elif meanCurdispx < 0:
                    Alldispmeannegx.append(meanCurdispx)
                    Alldispvarnegx.append(varCurdispx)
                    Timedispnegx.append(i * tcalibration)
                    
                    
        return ( Timespeed,
            Allspeedmean,
            Allspeedvar,Timeradius,
            Allradiusmean,
            Allradiusvar,Timedisppos,
            Alldispmeanpos,
            Alldispvarpos,Timedispneg,
            Alldispmeanneg,
            Alldispvarneg,Timedispposy,
            Alldispmeanposy,
            Alldispvarposy,Timedispnegy,
            Alldispmeannegy,
            Alldispvarnegy,Timedispposx,
            Alldispmeanposx,
            Alldispvarposx,Timedispnegx,
            Alldispmeannegx,
            Alldispvarnegx)            
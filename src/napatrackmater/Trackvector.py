from .Trackmate import TrackMate, get_feature_dict
from pathlib import Path
import lxml.etree as et
import concurrent
import os
import logging
import numpy as np
import re
import napari
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from typing import List, Union
import random
from tifffile import imwrite, imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, ccf
from scipy.stats import norm, anderson
from kapoorlabs_lightning.lightning_trainer import MitosisInception
from natsort import natsorted
import seaborn as sns
from tqdm import tqdm
import imageio
from skimage import measure
from matplotlib import cm
from IPython.display import clear_output
import torch
from collections import Counter
import csv
import h5py

logger = logging.getLogger(__name__)


SHAPE_FEATURES = [
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Local_Cell_Density",
    "Surface_Area",
]

DYNAMIC_FEATURES = [
    "Speed",
    "Motion_Angle_Z",
    "Motion_Angle_Y",
    "Motion_Angle_X",
    "Acceleration",
    "Distance_Cell_mask",
    "Radial_Angle_Z",
    "Radial_Angle_Y",
    "Radial_Angle_X",
    "Cell_Axis_Z",
    "Cell_Axis_Y",
    "Cell_Axis_X",
]

BROWNIAN_FEATURES = [
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Surface_Area",
    "Speed",
    "Acceleration",
]

TRACK_TYPE_FEATURES = ["MSD"]
IDENTITY_FEATURES = ["Track ID", "t", "z", "y", "x", "Dividing", "Number_Dividing"]

STATUS_FEATURES = ["Dividing", "Number_Dividing"]

SHAPE_DYNAMIC_FEATURES = SHAPE_FEATURES + DYNAMIC_FEATURES

ALL_FEATURES = IDENTITY_FEATURES + SHAPE_DYNAMIC_FEATURES + TRACK_TYPE_FEATURES


class TrackVector(TrackMate):
    def __init__(
        self,
        master_xml_path: Path,
        viewer=None,
        image: np.ndarray = None,
        spot_csv_path: Path = None,
        track_csv_path: Path = None,
        edges_csv_path: Path = None,
        t_minus: int = 0,
        t_plus: int = 10,
        x_start: int = 0,
        x_end: int = 10,
        y_start: int = 0,
        y_end: int = 10,
        show_tracks: bool = False,
        autoencoder_model=None,
        num_points=2048,
        latent_features=0,
        batch_size=1,
        scale_z=1.0,
        scale_xy=1.0,
        seg_image: np.ndarray = None,
        accelerator: str = "cpu",
        devices: Union[List[int], str, int] = 1,
    ):

        super().__init__(
            None,
            spot_csv_path,
            track_csv_path,
            edges_csv_path,
            image=image,
            seg_image=seg_image,
            AttributeBoxname="AttributeIDBox",
            TrackAttributeBoxname="TrackAttributeIDBox",
            TrackidBox="All",
            axes="TZYX",
            master_xml_path=None,
            autoencoder_model=autoencoder_model,
            scale_z=scale_z,
            scale_xy=scale_xy,
            latent_features=latent_features,
            accelerator=accelerator,
            devices=devices,
            num_points=num_points,
            batch_size=batch_size,
        )
        self._viewer = viewer
        self._image = image
        self.master_xml_path = master_xml_path
        self.spot_csv_path = spot_csv_path
        self.track_csv_path = track_csv_path
        self.edges_csv_path = edges_csv_path
        self._t_minus = t_minus
        self._t_plus = t_plus
        self._x_start = x_start
        self._x_end = x_end
        self._y_start = y_start
        self._y_end = y_end
        self._show_tracks = show_tracks
        self.autoencoder_model = autoencoder_model
        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self.accelerator = accelerator
        self.devices = devices
        self.num_points = num_points
        self.batch_size = batch_size
        self.seg_image = seg_image
        self.latent_features = latent_features
        self.xml_parser = et.XMLParser(huge_tree=True)

        self.unique_morphology_dynamic_properties = {}
        self.unique_mitosis_label = {}
        self.non_unique_mitosis_label = {}

        if not isinstance(self.master_xml_path, str):
            if self.master_xml_path.is_file():
                print("Reading Master XML")

                self.xml_content = et.fromstring(
                    open(self.master_xml_path).read().encode(), self.xml_parser
                )

                self.filtered_track_ids = [
                    int(track.get(self.trackid_key))
                    for track in self.xml_content.find("Model")
                    .find("FilteredTracks")
                    .findall("TrackID")
                ]
                self.max_track_id = max(self.filtered_track_ids)

                self._get_track_vector_xml_data()

    @property
    def viewer(self):
        return self._viewer

    @viewer.setter
    def viewer(self, value):
        self._viewer = value

    @property
    def x_start(self):
        return self._x_start

    @x_start.setter
    def x_start(self, value):
        self._x_start = value

    @property
    def y_start(self):
        return self._y_start

    @y_start.setter
    def y_start(self, value):
        self._y_start = value

    @property
    def x_end(self):
        return self._x_end

    @x_end.setter
    def x_end(self, value):
        self._x_end = value

    @property
    def y_end(self):
        return self._y_end

    @y_end.setter
    def y_end(self, value):
        self._y_end = value

    @property
    def t_minus(self):
        return self._t_minus

    @t_minus.setter
    def t_minus(self, value):
        self._t_minus = value

    @property
    def t_plus(self):
        return self._t_plus

    @t_plus.setter
    def t_plus(self, value):
        self._t_plus = value

    @property
    def show_tracks(self):
        return self._show_tracks

    @show_tracks.setter
    def show_tracks(self, value):
        self._show_tracks = value

    def _get_track_vector_xml_data(self):

        self.unique_objects = {}
        self.unique_properties = {}
        self.AllTrackIds = []
        self.DividingTrackIds = []
        self.NormalTrackIds = []
        self.GobletTrackIds = []
        self.BasalTrackIds = []
        self.RadialTrackIds = []
        self.all_track_properties = []
        self.split_points_times = []

        self.AllTrackIds.append(None)
        self.DividingTrackIds.append(None)
        self.NormalTrackIds.append(None)
        self.GobletTrackIds.append(None)
        self.BasalTrackIds.append(None)
        self.RadialTrackIds.append(None)

        self.AllTrackIds.append(self.TrackidBox)
        self.DividingTrackIds.append(self.TrackidBox)
        self.NormalTrackIds.append(self.TrackidBox)
        self.GobletTrackIds.append(self.TrackidBox)
        self.BasalTrackIds.append(self.TrackidBox)
        self.RadialTrackIds.append(self.TrackidBox)

        self.Spotobjects = self.xml_content.find("Model").find("AllSpots")
        # Extract the tracks from xml
        self.tracks = self.xml_content.find("Model").find("AllTracks")
        self.settings = self.xml_content.find("Settings").find("ImageData")
        self.xcalibration = float(self.settings.get("pixelwidth"))
        self.ycalibration = float(self.settings.get("pixelheight"))
        self.zcalibration = float(self.settings.get("voxeldepth"))
        self.tcalibration = int(float(self.settings.get("timeinterval")))
        self.detectorsettings = self.xml_content.find("Settings").find(
            "DetectorSettings"
        )
        self.basicsettings = self.xml_content.find("Settings").find("BasicSettings")
        try:
            self.detectorchannel = int(
                float(self.detectorsettings.get("TARGET_CHANNEL"))
            )
        except TypeError:
            self.detectorchannel = 1
        self.tstart = int(float(self.basicsettings.get("tstart")))
        self.tend = int(float(self.basicsettings.get("tend")))
        self.xmin = int(float(self.basicsettings.get("xstart")))
        self.xmax = int(float(self.basicsettings.get("xend")))
        self.ymin = int(float(self.basicsettings.get("ystart")))
        self.ymax = int(float(self.basicsettings.get("yend")))

        if self.x_end > self.xmax:
            self.x_end = self.xmax
        if self.y_end > self.ymax:
            self.y_end = self.ymax

        if self.x_start < self.xmin:
            self.x_start = self.xmin
        if self.y_start < self.ymin:
            self.y_start = self.ymin

    def _compute_track_vectors(self):

        self.current_shape_dynamic_vectors = []
        for k in self.unique_dynamic_properties.keys():
            dividing, number_dividing = self.unique_track_mitosis_label[k]
            nested_unique_dynamic_properties = self.unique_dynamic_properties[k]
            nested_unique_shape_properties = self.unique_shape_properties[k]
            for current_unique_id in nested_unique_dynamic_properties.keys():

                unique_dynamic_properties_tracklet = nested_unique_dynamic_properties[
                    current_unique_id
                ]
                (
                    current_time,
                    speed,
                    motion_angle_z,
                    motion_angle_y,
                    motion_angle_x,
                    acceleration,
                    distance_cell_mask,
                    radial_angle_z,
                    radial_angle_y,
                    radial_angle_x,
                    cell_axis_z,
                    cell_axis_y,
                    cell_axis_x,
                    _,
                    _,
                    _,
                    _,
                    msd,
                ) = unique_dynamic_properties_tracklet
                unique_shape_properties_tracklet = nested_unique_shape_properties[
                    current_unique_id
                ]
                (
                    current_time,
                    current_z,
                    current_y,
                    current_x,
                    radius,
                    eccentricity_comp_first,
                    eccentricity_comp_second,
                    eccentricity_comp_third,
                    local_cell_density,
                    surface_area,
                    latent_features,
                ) = unique_shape_properties_tracklet

                track_id_array = np.ones(current_time.shape)
                dividing_array = np.ones(current_time.shape)
                number_dividing_array = np.ones(current_time.shape)
                for i in range(track_id_array.shape[0]):
                    track_id_array[i] = track_id_array[i] * current_unique_id
                    dividing_array[i] = dividing_array[i] * dividing
                    number_dividing_array[i] = (
                        number_dividing_array[i] * number_dividing
                    )
                self.current_shape_dynamic_vectors.append(
                    [
                        track_id_array,
                        current_time,
                        current_z,
                        current_y,
                        current_x,
                        dividing_array,
                        number_dividing_array,
                        radius,
                        eccentricity_comp_first,
                        eccentricity_comp_second,
                        eccentricity_comp_third,
                        local_cell_density,
                        surface_area,
                        speed,
                        motion_angle_z,
                        motion_angle_y,
                        motion_angle_x,
                        acceleration,
                        distance_cell_mask,
                        radial_angle_z,
                        radial_angle_y,
                        radial_angle_x,
                        cell_axis_z,
                        cell_axis_y,
                        cell_axis_x,
                        msd,
                    ]
                    + (
                        [latent_features[i] for i in range(len(latent_features))]
                        if len(latent_features) > 0
                        else []
                    )
                )

        print(
            f"returning shape and dynamic vectors as list {len(self.current_shape_dynamic_vectors)}"
        )

    def _interactive_function(self):

        print("Iterating over spots in frame")
        self.count = 0
        futures = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:

            for frame in self.Spotobjects.findall("SpotsInFrame"):
                futures.append(executor.submit(self._master_spot_computer, frame))

            [r.result() for r in concurrent.futures.as_completed(futures)]

        print(f"Iterating over tracks {len(self.filtered_track_ids)}")
        self._correct_track_status()
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:

            for track in self.tracks.findall("Track"):

                track_id = int(track.get(self.trackid_key))
                if track_id in self.filtered_track_ids:
                    futures.append(
                        executor.submit(
                            self._master_track_computer,
                            track,
                            track_id,
                            self.t_minus,
                            self.t_plus,
                        )
                    )

            [r.result() for r in concurrent.futures.as_completed(futures)]
        self._get_cell_fate_tracks()
        print("getting attributes")
        if (
            self.spot_csv_path is not None
            and self.track_csv_path is not None
            and self.edges_csv_path is not None
        ):
            self._get_attributes()
        if self.autoencoder_model is not None:
            self._compute_latent_space()
        self.unique_tracks = {}
        self.unique_track_properties = {}
        self.unique_fft_properties = {}
        self.unique_cluster_properties = {}
        self.unique_shape_properties = {}
        self.unique_dynamic_properties = {}

        for track_id in self.filtered_track_ids:

            self._final_morphological_dynamic_vectors(track_id)
        self._compute_phenotypes()
        self._compute_track_vectors()

        if self._show_tracks:

            if len(list(self._viewer.layers)) > 0:
                layer_types = []
                for layer in list(self._viewer.layers):
                    layer_types.append(type(layer))

                if napari.layers.Image not in layer_types:
                    self._viewer.add_image(self._image)
            else:

                self._viewer.add_image(self._image)

            if len(self.unique_tracks.keys()) > 0:
                unique_tracks = np.concatenate(
                    [
                        self.unique_tracks[unique_track_id]
                        for unique_track_id in self.unique_tracks.keys()
                    ]
                )

                unique_tracks_properties = np.concatenate(
                    [
                        self.unique_track_properties[unique_track_id]
                        for unique_track_id in self.unique_track_properties.keys()
                    ]
                )
                features = get_feature_dict(unique_tracks_properties)

                for layer in list(self._viewer.layers):
                    if (
                        "Track" == layer.name
                        or "Boxes" == layer.name
                        or "Track_points" == layer.name
                    ):
                        self._viewer.layers.remove(layer)
                    vertices = unique_tracks[:, 1:]
                self._viewer.add_points(vertices, name="Track_points", size=1)
                self._viewer.add_tracks(unique_tracks, name="Track", features=features)

    def _final_morphological_dynamic_vectors(self, track_id):

        current_cell_ids = self.all_current_cell_ids[int(track_id)]
        current_tracklets = {}
        current_tracklets_properties = {}

        for i in range(len(current_cell_ids)):

            k = int(current_cell_ids[i])
            all_dict_values = self.unique_spot_properties[k]
            unique_id = str(all_dict_values[self.uniqueid_key])
            current_track_id = str(all_dict_values[self.trackid_key])
            t = int(float(all_dict_values[self.frameid_key]))
            z = float(all_dict_values[self.zposid_key])
            y = float(all_dict_values[self.yposid_key])
            x = float(all_dict_values[self.xposid_key])
            if (
                t >= self._t_minus
                and t <= self._t_plus
                and x >= self._x_start
                and x <= self._x_end
                and y >= self._y_start
                and y <= self._y_end
            ):
                (
                    current_tracklets,
                    current_tracklets_properties,
                ) = self._tracklet_and_properties(
                    all_dict_values,
                    t,
                    z,
                    y,
                    x,
                    k,
                    current_track_id,
                    unique_id,
                    current_tracklets,
                    current_tracklets_properties,
                )

        if str(track_id) in current_tracklets:
            current_tracklets = np.asarray(current_tracklets[str(track_id)])
            current_tracklets_properties = np.asarray(
                current_tracklets_properties[str(track_id)]
            )
            if len(current_tracklets.shape) == 2:
                self.unique_tracks[track_id] = current_tracklets
                self.unique_track_properties[track_id] = current_tracklets_properties

    def plot_mitosis_times(self, full_dataframe, save_path=""):

        subset = full_dataframe[full_dataframe["Dividing"] == 1].loc[
            full_dataframe.duplicated(subset=["t", "x", "y", "z"], keep=False)
        ]

        dividing_counts = subset.groupby("t").size() / 2
        times = dividing_counts.index
        counts = dividing_counts.values
        data = {"Time": times, "Count": counts}
        df = pd.DataFrame(data)
        np.save(save_path + "counts.npy", df.to_numpy())

        max_number_dividing = full_dataframe["Number_Dividing"].max()
        min_number_dividing = full_dataframe["Number_Dividing"].min()
        excluded_keys = [
            "Track ID",
            "t",
            "z",
            "y",
            "x",
            "Unnamed: 0",
            "Unnamed",
            "Track Duration",
            "Generation ID",
            "TrackMate Track ID",
            "Tracklet Number ID",
        ]
        for i in range(
            min_number_dividing.astype(int), max_number_dividing.astype(int) + 1
        ):
            for column in full_dataframe.columns:
                if column not in excluded_keys:
                    data = full_dataframe[column][
                        full_dataframe["Number_Dividing"].astype(int) == i
                    ]

                    np.save(
                        f"{save_path}{column}Number_Dividing_{i}.npy", data.to_numpy()
                    )

        all_split_data = []

        for split_id in tqdm(self.split_cell_ids, desc="Cell split IDs"):
            spot_properties = self.unique_spot_properties[split_id]
            track_id = spot_properties[self.trackid_key]
            unique_id = spot_properties[self.uniqueid_key]
            tracklet_id = spot_properties[self.trackletid_key]
            number_times_divided = spot_properties[self.number_dividing_key]
            surface_area = spot_properties[self.surface_area_key]
            eccentricity_comp_first = spot_properties[self.eccentricity_comp_firstkey]
            eccentricity_comp_second = spot_properties[self.eccentricity_comp_secondkey]
            eccentricity_comp_third = spot_properties[self.eccentricity_comp_thirdkey]
            local_cell_density = spot_properties[self.local_cell_density_key]
            radius = spot_properties[self.radius_key]
            speed = spot_properties[self.speed_key]

            motion_angle_z = spot_properties[self.motion_angle_z_key]
            motion_angle_y = spot_properties[self.motion_angle_y_key]
            motion_angle_x = spot_properties[self.motion_angle_x_key]
            acceleration = spot_properties[self.acceleration_key]
            distance_cell_mask = spot_properties[self.distance_cell_mask_key]
            radial_angle_z = spot_properties[self.radial_angle_z_key]
            radial_angle_y = spot_properties[self.radial_angle_y_key]
            radial_angle_x = spot_properties[self.radial_angle_x_key]
            cell_axis_z = spot_properties[self.cell_axis_z_key]
            cell_axis_y = spot_properties[self.cell_axis_y_key]
            cell_axis_x = spot_properties[self.cell_axis_x_key]

            data = {
                "Track_ID": track_id,
                "Unique_ID": unique_id,
                "Tracklet_ID": tracklet_id,
                "Number_Times_Divided": number_times_divided,
                "Surface_Area": surface_area,
                "Eccentricity_Comp_First": eccentricity_comp_first,
                "Eccentricity_Comp_Second": eccentricity_comp_second,
                "Eccentricity_Comp_Third": eccentricity_comp_third,
                "Local_Cell_Density": local_cell_density,
                "Radius": radius,
                "Speed": speed,
                "Motion_Angle_Z": motion_angle_z,
                "Motion_Angle_Y": motion_angle_y,
                "Motion_Angle_X": motion_angle_x,
                "Acceleration": acceleration,
                "Distance_Cell_Mask": distance_cell_mask,
                "Radial_Angle_Z": radial_angle_z,
                "Radial_Angle_Y": radial_angle_y,
                "Radial_Angle_X": radial_angle_x,
                "cell_axis_Z": cell_axis_z,
                "cell_axis_Y": cell_axis_y,
                "cell_axis_X": cell_axis_x,
            }

            all_split_data.append(data)

        np.save(f"{save_path}data_at_mitosis_time.npy", all_split_data)

    def plot_cell_type_times(self, correlation_dataframe, save_path=""):

        cell_types = correlation_dataframe["Cell_Type_Label"].unique()

        excluded_keys = [
            "Track ID",
            "t",
            "z",
            "y",
            "x",
            "Unnamed: 0",
            "Unnamed",
            "Track Duration",
            "Generation ID",
            "TrackMate Track ID",
            "Tracklet Number ID",
            "Cell_Type",
        ]

        for ctype in cell_types:
            for column in correlation_dataframe.columns:
                if column not in excluded_keys:
                    data = correlation_dataframe[column][
                        correlation_dataframe["Cell_Type_Label"].astype(int) == ctype
                    ]

                    np.save(
                        f"{save_path}{column}Cell_Type_{ctype}.npy", data.to_numpy()
                    )

    def get_shape_dynamic_feature_dataframe(self):

        current_shape_dynamic_vectors = self.current_shape_dynamic_vectors
        tracks_dataframe = []

        for i in range(len(current_shape_dynamic_vectors)):
            vector_list = current_shape_dynamic_vectors[i]
            numel_features = len(ALL_FEATURES)
            initial_array = np.array(vector_list[:numel_features])
            latent_shape_features = np.array(vector_list[numel_features:])
            zipped_initial_array = list(zip(initial_array))
            data_frame_list = np.transpose(
                np.asarray(
                    [zipped_initial_array[i] for i in range(len(zipped_initial_array))]
                )[:, 0, :]
            )

            shape_dynamic_dataframe = pd.DataFrame(
                data_frame_list,
                columns=ALL_FEATURES,
            )

            if len(latent_shape_features) > 0:
                new_columns = [
                    f"latent_feature_number_{i}"
                    for i in range(latent_shape_features.shape[1])
                ]
                latent_features_df = pd.DataFrame(
                    latent_shape_features, columns=new_columns
                )

                shape_dynamic_dataframe = pd.concat(
                    [shape_dynamic_dataframe, latent_features_df], axis=1
                )

            if len(tracks_dataframe) == 0:
                tracks_dataframe = shape_dynamic_dataframe
            else:
                tracks_dataframe = pd.concat(
                    [tracks_dataframe, shape_dynamic_dataframe],
                    ignore_index=True,
                )
        tracks_dataframe["TrackMate Track ID"] = tracks_dataframe["Track ID"].map(
            self.tracklet_id_to_trackmate_id
        )

        tracks_dataframe["Generation ID"] = tracks_dataframe["Track ID"].map(
            self.tracklet_id_to_generation_id
        )

        tracks_dataframe["Tracklet Number ID"] = tracks_dataframe["Track ID"].map(
            self.tracklet_id_to_tracklet_number_id
        )

        trackmate_ids = tracks_dataframe["TrackMate Track ID"]
        track_duration_dict = {}
        for trackmate_id in trackmate_ids:
            track_properties = self.unique_track_properties[trackmate_id]
            total_track_duration = track_properties[:, -1][0]
            track_duration_dict[trackmate_id] = int(total_track_duration)
        tracks_dataframe["Track Duration"] = tracks_dataframe["TrackMate Track ID"].map(
            track_duration_dict
        )

        tracks_dataframe = tracks_dataframe.sort_values(by=["Track ID"])
        tracks_dataframe = tracks_dataframe.sort_values(by=["t"])

        return tracks_dataframe

    def build_closeness_dict(self, radius):
        self.closeness_dict = {}

        unique_time_points = set()
        for track_data in self.unique_tracks.values():
            for entry in track_data:
                t, z, y, x = entry[1], entry[2], entry[3], entry[4]
                unique_time_points.update(int(float(t)))

        for time_point in unique_time_points:
            coords = []
            track_ids = []

            for track_id, track_data in self.unique_tracks.items():
                for entry in track_data:
                    t, z, y, x = entry[1], entry[2], entry[3], entry[4]
                    if t == time_point:
                        coords.append([z, y, x])
                        track_ids.append(track_id)

            coords = np.array(coords)
            track_ids = np.array(track_ids)

            tree = cKDTree(coords)

            closeness_dict_time_point = {}
            for track_id, track_data in self.unique_tracks.items():
                for entry in track_data:
                    t, z, y, x = entry[1], entry[2], entry[3], entry[4]
                    if t == time_point:
                        closest_indices = tree.query_ball_point((z, y, x), r=radius)
                        closest_track_ids = track_ids[np.concatenate(closest_indices)]
                        closeness_dict_time_point[track_id] = list(closest_track_ids)

            self.closeness_dict[time_point] = closeness_dict_time_point


def _iterate_over_tracklets(
    track_data, training_tracklets, track_id, prediction=False, ignore_columns=[]
):

    shape_dynamic_dataframe = track_data[SHAPE_DYNAMIC_FEATURES].copy()

    shape_dataframe = track_data[SHAPE_FEATURES].copy()

    dynamic_dataframe = track_data[DYNAMIC_FEATURES].copy()
    if not prediction:
        full_dataframe = track_data[ALL_FEATURES].copy()
    else:
        full_dataframe = track_data[ALL_FEATURES - STATUS_FEATURES].copy()

    if ignore_columns is not None:
        for column in ignore_columns:
            if column in full_dataframe.columns:
                full_dataframe.drop(columns=[column], inplace=True)
            if column in shape_dynamic_dataframe.columns:
                shape_dynamic_dataframe.drop(columns=[column], inplace=True)
            if column in shape_dataframe.columns:
                shape_dataframe.drop(columns=[column], inplace=True)
            if column in dynamic_dataframe.columns:
                dynamic_dataframe.drop(columns=[column], inplace=True)

    latent_columns = [
        col for col in track_data.columns if col.startswith("latent_feature_number_")
    ]
    if latent_columns:
        latent_features = track_data[latent_columns].copy()
        full_dataframe = pd.concat([full_dataframe, latent_features], axis=1)
        shape_dataframe = pd.concat([shape_dataframe, latent_features], axis=1)
        shape_dynamic_dataframe = pd.concat(
            [shape_dynamic_dataframe, latent_features], axis=1
        )

    # Drop rows with NaN values
    shape_dynamic_dataframe.dropna(inplace=True)
    shape_dataframe.dropna(inplace=True)
    dynamic_dataframe.dropna(inplace=True)
    full_dataframe.dropna(inplace=True)

    shape_dynamic_dataframe_list = shape_dynamic_dataframe.to_dict(orient="records")
    shape_dataframe_list = shape_dataframe.to_dict(orient="records")
    dynamic_dataframe_list = dynamic_dataframe.to_dict(orient="records")
    full_dataframe_list = full_dataframe.to_dict(orient="records")
    training_tracklets[track_id] = (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
    )

    return training_tracklets


def _iterate_over_cell_type_tracklets(
    track_data,
    training_tracklets,
    track_id,
    prediction=False,
    ignore_columns=[],
    cell_type="Cell_Type_Label",
):

    shape_dynamic_dataframe = track_data[SHAPE_DYNAMIC_FEATURES].copy()

    shape_dataframe = track_data[SHAPE_FEATURES].copy()

    dynamic_dataframe = track_data[DYNAMIC_FEATURES].copy()

    cell_type_track = int(float(track_data[cell_type].iloc[0]))

    if not prediction:
        full_dataframe = track_data[ALL_FEATURES].copy()
    else:
        full_dataframe = track_data[ALL_FEATURES - STATUS_FEATURES].copy()

    if ignore_columns is not None:
        for column in ignore_columns:
            if column in full_dataframe.columns:
                full_dataframe.drop(columns=[column], inplace=True)
            if column in shape_dynamic_dataframe.columns:
                shape_dynamic_dataframe.drop(columns=[column], inplace=True)
            if column in shape_dataframe.columns:
                shape_dataframe.drop(columns=[column], inplace=True)
            if column in dynamic_dataframe.columns:
                dynamic_dataframe.drop(columns=[column], inplace=True)

    latent_columns = [
        col for col in track_data.columns if col.startswith("latent_feature_number_")
    ]
    if latent_columns:
        latent_features = track_data[latent_columns].copy()
        full_dataframe = pd.concat([full_dataframe, latent_features], axis=1)
        shape_dataframe = pd.concat([shape_dataframe, latent_features], axis=1)
        shape_dynamic_dataframe = pd.concat(
            [shape_dynamic_dataframe, latent_features], axis=1
        )

    # Drop rows with NaN values
    shape_dynamic_dataframe.dropna(inplace=True)
    shape_dataframe.dropna(inplace=True)
    dynamic_dataframe.dropna(inplace=True)
    full_dataframe.dropna(inplace=True)

    shape_dynamic_dataframe_list = shape_dynamic_dataframe.to_dict(orient="records")
    shape_dataframe_list = shape_dataframe.to_dict(orient="records")
    dynamic_dataframe_list = dynamic_dataframe.to_dict(orient="records")
    full_dataframe_list = full_dataframe.to_dict(orient="records")
    training_tracklets[track_id] = (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
        cell_type_track,
    )

    return training_tracklets


def create_dividing_prediction_tracklets(
    tracks_dataframe: pd.DataFrame, ignore_columns=[]
):
    training_tracklets = {}
    subset_dividing = tracks_dataframe[tracks_dataframe["Dividing"] == 1]
    track_ids = subset_dividing["Track ID"].unique()
    for track_id in track_ids:
        track_data = tracks_dataframe[
            (tracks_dataframe["Track ID"] == track_id)
        ].sort_values(by="t")
        if track_data.shape[0] > 0:
            training_tracklets = _iterate_over_tracklets(
                track_data,
                training_tracklets,
                track_id,
                prediction=True,
                ignore_columns=ignore_columns,
            )

    return training_tracklets


def filter_and_get_tracklets(
    df,
    cell_type,
    N,
    raw_image,
    crop_size,
    segmentation_image,
    dataset_name,
    save_dir,
    train_label,
    class_map_gbr={0: "Basal", 1: "Radial", 2: "Goblet"},
):

    """
    Filters tracklets from a DataFrame based on a specified cell type, extracts blocks of a specified length (N) from each tracklet,
    and generates a volume around each tracklet block based on the segmentation image.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing tracking information with columns `TrackMate Track ID`, `Track ID`, `t`, `z`, `y`, `x`, and `Cell_Type`.

    cell_type : str
        The cell type to filter by (e.g., "Basal", "Radial", "Goblet").

    N : int
        The number of timepoints to include in each tracklet block.

    raw_image : ndarray
        The raw image volume for the dataset (assumed format: TZYX).

    crop_size : tuple
        The dimensions (sizex, sizey, sizez) for each cropped patch.

    segmentation_image : ndarray
        Segmentation image used for calculating the bounding box around each tracklet point (assumed format: TZYX).

    dataset_name : str
        Name of the dataset, used for labeling output files.

    save_dir : str
        Directory to save cropped patches and related data.

    train_label : int
        Label for the training category, based on `cell_type`.

    normalize_image : bool, optional, default=True
        Whether to normalize the raw image data before generating volumes.

    dtype : dtype, optional, default=np.float32
        Data type for the processed image.

    class_map_gbr : dict, optional, default={0: "Basal", 1: "Radial", 2: "Goblet"}
        Mapping from numerical labels to cell types.

    Returns:
    -------
    tracklets : dict
        Nested dictionary with `TrackMate Track ID` as keys, each containing dictionaries
        with `Track ID` keys and lists of `N`-sized tracklet blocks as values.

    Notes:
    ------
    - The function saves cropped patches centered around each tracklet block's (t, z, y, x) coordinates.
    - Each block is extracted for exactly `N` timepoints, sorted in ascending order by time.
    """

    total_categories = len(class_map_gbr)
    cell_type_df = df[df["Cell_Type"] == cell_type]
    tracklets = {}

    for trackmate_id in cell_type_df["TrackMate Track ID"].unique():
        trackmate_df = cell_type_df[cell_type_df["TrackMate Track ID"] == trackmate_id]

        for track_id in trackmate_df["Track ID"].unique():
            track_df = trackmate_df[trackmate_df["Track ID"] == track_id]

            track_df = track_df.sort_values(by="t")

            tracklet_blocks = []
            for i in range(0, len(track_df), N):
                tracklet_block = track_df.iloc[i : i + N][["t", "z", "y", "x"]].values
                if len(tracklet_block) == N:
                    tracklet_blocks.append(tracklet_block)

            if trackmate_id not in tracklets:
                tracklets[trackmate_id] = {}
            tracklets[trackmate_id][track_id] = tracklet_blocks

    for trackmate_id in tracklets:
        for track_id in tracklets[trackmate_id]:
            for idx, tracklet_block in enumerate(tracklets[trackmate_id][track_id]):

                name = (
                    dataset_name
                    + str(trackmate_id)
                    + str(int(track_id))
                    + str(idx)
                    + "_"
                    + str(cell_type)
                )
                TrackVolumeMaker(
                    tracklet_block,
                    raw_image,
                    segmentation_image,
                    crop_size,
                    total_categories,
                    train_label,
                    name,
                    save_dir,
                )

    return tracklets


def getHWD(
    defaultX,
    defaultY,
    defaultZ,
    currentsegimage,
):

    properties = measure.regionprops(currentsegimage)
    SegLabel = currentsegimage[int(defaultZ), int(defaultY), int(defaultX)]

    for prop in properties:
        if SegLabel > 0 and prop.label == SegLabel:
            minr, minc, mind, maxr, maxc, maxd = prop.bbox
            center = (defaultZ, defaultY, defaultX)
            height = abs(maxc - minc)
            width = abs(maxr - minr)
            depth = abs(maxd - mind)
            return height, width, depth, center, SegLabel


def normalizeFloatZeroOne(x, pmin=1, pmax=99.8, axis=None, eps=1e-20, dtype=np.uint8):
    """Percentile based Normalization

    Normalize patches of image before feeding into the network

    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, eps=1e-20, dtype=np.uint8):

    x = x.astype(dtype)
    mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
    ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
    eps = dtype(eps) if np.isscalar(eps) else eps.astype(dtype, copy=False)

    x = (x - mi) / (ma - mi + eps)

    return x


def normalize_image_in_chunks(
    originalimage,
    chunk_steps=50,
    percentile_min=1,
    percentile_max=99.8,
    dtype=np.float32,
):
    """
    Normalize a TZYX image in chunks along the T (time) dimension.

    Args:
        image (np.ndarray): The original TZYX image.
        chunk_size (int): The number of timesteps to process at a time.
        percentile_min (float): The lower percentile for normalization.
        percentile_max (float): The upper percentile for normalization.
        dtype (np.dtype): The data type to cast the normalized image.

    Returns:
        np.ndarray: The normalized image with the same shape as the input.
    """

    # Get the shape of the original image (T, Z, Y, X)
    T, Z, Y, X = originalimage.shape

    # Create an empty array to hold the normalized image
    normalized_image = np.empty((T, Z, Y, X), dtype=dtype)

    # Process the image in chunks of `chunk_size` along the T (time) axis
    for t in range(0, T, chunk_steps):
        # Determine the chunk slice, ensuring we don't go out of bounds
        t_end = min(t + chunk_steps, T)

        # Extract the chunk of timesteps to normalize
        chunk = originalimage[t:t_end]

        # Normalize this chunk
        chunk_normalized = normalizeFloatZeroOne(
            chunk, percentile_min, percentile_max, dtype=dtype
        )

        # Replace the corresponding portion of the original image with the normalized chunk
        normalized_image[t:t_end] = chunk_normalized

    return normalized_image


def TrackVolumeMaker(
    tracklet_block,
    raw_image,
    segmentation_image,
    crop_size,
    total_categories,
    train_label,
    name,
    save_dir,
):
    sizex, sizey, sizez = crop_size

    stitched_volume = []
    for idx, (t, z, y, x) in enumerate(tracklet_block):
        # Get the bounding box properties based on segmentation image
        current_seg_image = segmentation_image[int(t)].astype("uint16")
        image_props = getHWD(x, y, z, current_seg_image)

        if image_props is not None:
            height, width, depth, center, seg_label = image_props
            small_image = raw_image[int(t)]
            x = center[2]
            y = center[1]
            z = center[0]
            if height >= sizey:
                height = 0.5 * sizey
            if width >= sizex:
                width = 0.5 * sizex
            if depth >= sizez:
                depth = 0.5 * sizez

            if (
                x > sizex / 2
                and z > sizez / 2
                and y > sizey / 2
                and z + int(sizez / 2) < raw_image.shape[1]
                and y + int(sizey / 2) < raw_image.shape[2]
                and x + int(sizex / 2) < raw_image.shape[3]
                and t < raw_image.shape[0]
            ):
                crop_xminus = x - int(sizex / 2)
                crop_xplus = x + int(sizex / 2)
                crop_yminus = y - int(sizey / 2)
                crop_yplus = y + int(sizey / 2)
                crop_zminus = z - int(sizez / 2)
                crop_zplus = z + int(sizez / 2)
                region = (
                    slice(int(crop_zminus), int(crop_zplus)),
                    slice(int(crop_yminus), int(crop_yplus)),
                    slice(int(crop_xminus), int(crop_xplus)),
                )

                crop_image = small_image[region]
                if idx == len(tracklet_block) // 2:
                    seglocationx = center[2] - crop_xminus
                    seglocationy = center[1] - crop_yminus
                    seglocationz = center[0] - crop_zminus

                stitched_volume.append(crop_image)
    if len(stitched_volume) > 0:
        stitched_volume = np.stack(stitched_volume, axis=0)
        print(stitched_volume.shape, len(tracklet_block))
        if stitched_volume.shape[0] == len(tracklet_block):
            label_vector = np.zeros(total_categories + 7)
            label_vector[train_label] = 1
            label_vector[total_categories + 6] = 1
            label_vector[total_categories] = seglocationx / sizex
            label_vector[total_categories + 1] = seglocationy / sizey
            label_vector[total_categories + 2] = seglocationz / sizez
            label_vector[total_categories + 3] = height / sizey
            label_vector[total_categories + 4] = width / sizex
            label_vector[total_categories + 5] = depth / sizez

            volume_name = f"track_{name}_stitched_volume.tif"
            volume_path = os.path.join(save_dir, volume_name)
            label_name = f"track_{name}_stitched_volume.csv"
            label_path = os.path.join(save_dir, label_name)

            if not os.path.exists(volume_path) and not os.path.exists(label_path):

                imwrite(volume_path, stitched_volume.astype("float32"))

                with open(label_path, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(label_vector)

                print(
                    f"Saved stitched volume to {volume_path} and label to {label_path}"
                )


def create_h5(
    save_dir,
    train_size=0.95,
    save_name="cellfate_vision_training_data_gbr",
):
    """
    Create HDF5 file with training and validation data for morphodynamic model in TZYX format.

    Args:
        save_dir (str): Directory containing image and label files.
        train_size (float): Proportion of data to use for training.
        save_name (str): Name of the output HDF5 file (without extension).
    """
    data = []
    labels = []

    # Gather all TIFF files and their corresponding labels
    all_files = os.listdir(save_dir)
    tif_files = sorted([f for f in all_files if f.endswith(".tif")])

    # Load images and labels in T, Z, Y, X format
    for tif_file in tif_files:
        image_path = os.path.join(save_dir, tif_file)
        image = imread(image_path)

        # Ensure the image has T, Z, Y, X format, where T is the leading dimension
        if image.ndim != 4:
            raise ValueError(
                f"Image {tif_file} does not have four dimensions, expected T, Z, Y, X."
            )

        data.append(image)

        # Load corresponding label CSV
        csv_path = os.path.join(save_dir, os.path.splitext(tif_file)[0] + ".csv")
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            label = np.array(list(reader)[0]).astype(np.float32)
        labels.append(label)

    data = np.array(data)  # Shape: (N, T, Z, Y, X)
    labels = np.array(labels)  # Shape: (N, label_dim)

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, train_size=train_size, shuffle=True, random_state=42
    )

    h5_save_path = os.path.join(save_dir, f"{save_name}.h5")
    with h5py.File(h5_save_path, "w") as hf:
        hf.create_dataset("train_arrays", data=train_data)
        hf.create_dataset("train_labels", data=train_labels)
        hf.create_dataset("val_arrays", data=val_data)
        hf.create_dataset("val_labels", data=val_labels)

    print(f"HDF5 training data saved to {h5_save_path} in T, Z, Y, X format.")


def create_analysis_tracklets(
    tracks_dataframe: pd.DataFrame,
    t_minus=None,
    t_plus=None,
    class_ratio=-1,
    ignore_columns=[],
):
    training_tracklets = {}
    if t_minus is not None and t_plus is not None:
        time_mask = (tracks_dataframe["t"] >= t_minus) & (
            tracks_dataframe["t"] <= t_plus
        )
        local_shape_dynamic_dataframe = tracks_dataframe[time_mask]
    else:
        local_shape_dynamic_dataframe = tracks_dataframe

    subset_dividing = local_shape_dynamic_dataframe[
        local_shape_dynamic_dataframe["Dividing"] == 1
    ]

    subset_non_dividing = local_shape_dynamic_dataframe[
        local_shape_dynamic_dataframe["Dividing"] == 0
    ]
    non_dividing_track_ids = subset_non_dividing["Track ID"].unique()
    dividing_track_ids = subset_dividing["Track ID"].unique()
    dividing_count = len(dividing_track_ids)
    non_dividing_count = len(non_dividing_track_ids)
    if non_dividing_count > dividing_count and class_ratio > 0:
        non_dividing_track_ids = random.sample(
            list(non_dividing_track_ids),
            int(min(non_dividing_count, dividing_count * class_ratio)),
        )
    else:
        dividing_track_ids = random.sample(list(dividing_track_ids), dividing_count)
        non_dividing_track_ids = random.sample(
            list(non_dividing_track_ids), non_dividing_count
        )

    for track_id in dividing_track_ids:
        subset_dividing = subset_dividing.loc[
            local_shape_dynamic_dataframe.duplicated(
                subset=["t", "x", "y", "z"], keep=False
            )
        ]
        track_data = local_shape_dynamic_dataframe[
            (local_shape_dynamic_dataframe["Track ID"] == track_id)
        ].sort_values(by="t")
        if track_data.shape[0] > 0:
            training_tracklets = _iterate_over_tracklets(
                track_data, training_tracklets, track_id, ignore_columns=ignore_columns
            )

    for track_id in non_dividing_track_ids:
        track_data = local_shape_dynamic_dataframe[
            (local_shape_dynamic_dataframe["Track ID"] == track_id)
        ].sort_values(by="t")
        if track_data.shape[0] > 0:
            training_tracklets = _iterate_over_tracklets(
                track_data, training_tracklets, track_id, ignore_columns=ignore_columns
            )
    modified_dataframe = local_shape_dynamic_dataframe.copy()
    if ignore_columns is not None:
        for column in ignore_columns:
            if column in modified_dataframe.columns:
                modified_dataframe.drop(columns=[column], inplace=True)

    return training_tracklets, modified_dataframe


def create_analysis_cell_type_tracklets(
    tracks_dataframe: pd.DataFrame,
    t_minus=None,
    t_plus=None,
    class_ratio=-1,
    cell_type="Cell_Type_Labels",
    ignore_columns=[],
):
    training_tracklets = {}
    if t_minus is not None and t_plus is not None:
        time_mask = (tracks_dataframe["t"] >= t_minus) & (
            tracks_dataframe["t"] <= t_plus
        )
        local_shape_dynamic_dataframe = tracks_dataframe[time_mask]
    else:
        local_shape_dynamic_dataframe = tracks_dataframe

    subset_dividing = local_shape_dynamic_dataframe[
        local_shape_dynamic_dataframe["Dividing"] == 1
    ]

    subset_non_dividing = local_shape_dynamic_dataframe[
        local_shape_dynamic_dataframe["Dividing"] == 0
    ]
    non_dividing_track_ids = subset_non_dividing["Track ID"].unique()
    dividing_track_ids = subset_dividing["Track ID"].unique()
    dividing_count = len(dividing_track_ids)
    non_dividing_count = len(non_dividing_track_ids)
    if non_dividing_count > dividing_count and class_ratio > 0:
        non_dividing_track_ids = random.sample(
            list(non_dividing_track_ids),
            int(min(non_dividing_count, dividing_count * class_ratio)),
        )
    else:
        dividing_track_ids = random.sample(list(dividing_track_ids), dividing_count)
        non_dividing_track_ids = random.sample(
            list(non_dividing_track_ids), non_dividing_count
        )

    for track_id in dividing_track_ids:
        subset_dividing = subset_dividing.loc[
            local_shape_dynamic_dataframe.duplicated(
                subset=["t", "x", "y", "z"], keep=False
            )
        ]
        track_data = local_shape_dynamic_dataframe[
            (local_shape_dynamic_dataframe["Track ID"] == track_id)
        ].sort_values(by="t")
        if track_data.shape[0] > 0:
            training_tracklets = _iterate_over_cell_type_tracklets(
                track_data,
                training_tracklets,
                track_id,
                ignore_columns=ignore_columns,
                cell_type=cell_type,
            )

    for track_id in non_dividing_track_ids:
        track_data = local_shape_dynamic_dataframe[
            (local_shape_dynamic_dataframe["Track ID"] == track_id)
        ].sort_values(by="t")
        if track_data.shape[0] > 0:
            training_tracklets = _iterate_over_cell_type_tracklets(
                track_data,
                training_tracklets,
                track_id,
                ignore_columns=ignore_columns,
                cell_type=cell_type,
            )
    modified_dataframe = local_shape_dynamic_dataframe.copy()
    if ignore_columns is not None:
        for column in ignore_columns:
            if column in modified_dataframe.columns:
                modified_dataframe.drop(columns=[column], inplace=True)

    return training_tracklets, modified_dataframe


def create_gt_analysis_vectors_dict(tracks_dataframe: pd.DataFrame):
    gt_analysis_vectors = {}
    for track_id in tracks_dataframe["Track ID"].unique():
        track_data = tracks_dataframe[
            tracks_dataframe["Track ID"] == track_id
        ].sort_values(by="t")
        shape_dynamic_dataframe = track_data[SHAPE_DYNAMIC_FEATURES]
        gt_dataframe = track_data[
            [
                "Cluster",
            ]
        ]

        full_dataframe = track_data[ALL_FEATURES]

        latent_columns = [
            col
            for col in track_data.columns
            if col.startswith("latent_feature_number_")
        ]
        if latent_columns:
            latent_features = track_data[latent_columns]
            for col in latent_features.columns:
                full_dataframe[col] = latent_features[col]
                gt_dataframe[col] = latent_features[col]

        shape_dynamic_dataframe_list = shape_dynamic_dataframe.to_dict(orient="records")
        gt_dataframe_list = gt_dataframe.to_dict(orient="records")
        full_dataframe_list = full_dataframe.to_dict(orient="records")
        gt_analysis_vectors[track_id] = (
            shape_dynamic_dataframe_list,
            gt_dataframe_list,
            full_dataframe_list,
        )

    return gt_analysis_vectors


def create_global_gt_dataframe(
    full_dataframe,
    ground_truth_csv_file,
    calibration_z,
    calibration_y,
    calibration_x,
    time_veto_threshold=0.0,
    space_veto_threshold=5.0,
    cell_type_key="Celltype_label",
):

    ground_truth_data_frame = pd.read_csv(ground_truth_csv_file)
    ground_truth_data_frame.dropna(subset=[cell_type_key], inplace=True)

    # Prepare ground truth tuples and labels
    ground_truth_tuples = np.array(
        [
            ground_truth_data_frame["FRAME"].values,
            ground_truth_data_frame["POSITION_Z"].values / calibration_z,
            ground_truth_data_frame["POSITION_Y"].values / calibration_y,
            ground_truth_data_frame["POSITION_X"].values / calibration_x,
        ]
    ).T
    ground_truth_labels = ground_truth_data_frame[cell_type_key].values
    theory_tuples_spatial = full_dataframe[["t", "z", "y", "x"]].values

    tree_spatial = cKDTree(theory_tuples_spatial)

    # Initialize arrays to store the indices of the closest theory tuples and their corresponding ground truth labels
    closest_theory_indices = []
    corresponding_ground_truth_labels = []
    closest_theory_tuples_found = []
    closest_theory_track_ids = []
    # Find the closest theory tuple for each ground truth tuple
    for i, (ground_tuple, ground_label) in enumerate(
        zip(ground_truth_tuples, ground_truth_labels)
    ):
        # Use the KD-Tree for nearest-neighbor search
        spatial_valid_indices = tree_spatial.query_ball_point(
            ground_tuple, space_veto_threshold, p=2
        )

        if spatial_valid_indices:
            # Find the closest theory index within the common indices
            closest_theory_index = spatial_valid_indices[
                np.argmin(tree_spatial.query(ground_tuple)[0])
            ]

            # Get the closest theory tuple
            closest_theory_tuple = full_dataframe.loc[closest_theory_index]

            # Check if the index is valid, within the DataFrame's range, and satisfies the time veto
            if (
                0 <= closest_theory_index < len(full_dataframe)
                and abs(closest_theory_tuple["t"] - ground_tuple[0])
                <= time_veto_threshold
            ):
                closest_theory_indices.append(closest_theory_index)
                corresponding_ground_truth_labels.append(ground_label)
                closest_theory_tuples_found.append(closest_theory_tuple)
                closest_theory_track_ids.append(closest_theory_tuple["Track ID"])

    track_id_to_cluster = {
        track_id: cluster_label
        for track_id, cluster_label in zip(
            closest_theory_track_ids, corresponding_ground_truth_labels
        )
    }
    full_dataframe["Cluster"] = full_dataframe["Track ID"].map(track_id_to_cluster)

    return full_dataframe


def calculate_wcss(data, labels, centroids):
    wcss = 0
    for i in range(len(data)):
        cluster_label = labels[i]
        centroid = centroids[cluster_label]
        distance = np.linalg.norm(data[i] - centroid)
        wcss += distance**2
    return wcss


def calculate_intercluster_distance(compute_vectors, labels):
    intercluster_distances = {}
    for cluster_label in np.unique(labels):
        cluster_indices = np.where(labels == cluster_label)[0]

        compute_data = compute_vectors[cluster_indices]
        mean_vector = np.mean(compute_data, axis=0)

        distances = np.linalg.norm(compute_data - mean_vector, axis=1)
        mean_distance = np.mean(distances)

        intercluster_distances[cluster_label] = mean_distance
    return intercluster_distances


def calculate_intercluster_eucledian_distance(compute_vectors, labels):

    intercluster_eucledian_distances = {}
    for cluster_label in np.unique(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        compute_data = compute_vectors[cluster_indices]
        mean_vector = np.mean(compute_data, axis=0)
        distances = np.linalg.norm(compute_data - mean_vector, axis=1)
        mean_distance = np.mean(distances)
        intercluster_eucledian_distances[cluster_label] = mean_distance

    return intercluster_eucledian_distances


def calculate_cluster_centroids(data, labels):
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        cluster_data = data[labels == label]

        centroid = np.mean(cluster_data, axis=0)
        centroids[label] = centroid
    return centroids


def simple_unsupervised_clustering(
    full_dataframe,
    csv_file_name,
    analysis_vectors,
    cluster_threshold_shape_dynamic=3,
    cluster_threshold_dynamic=3,
    cluster_threshold_shape=3,
    t_delta=10,
    metric="euclidean",
    method="centroid",
    criterion="distance",
    distance_vectors="shape",
):

    analysis_track_ids = []
    shape_dynamic_covariance_matrix = []
    position_matrix = []
    shape_covariance_matrix = []
    dynamic_covariance_matrix = []
    for track_id, (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
    ) in analysis_vectors.items():
        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )
        shape_track_array = np.array(
            [[item for item in record.values()] for record in shape_dataframe_list]
        )
        dynamic_track_array = np.array(
            [[item for item in record.values()] for record in dynamic_dataframe_list]
        )
        columns_of_interest = ["t", "z", "y", "x"]
        position_track_array = np.array(
            [
                [record[col] for col in columns_of_interest]
                for record in full_dataframe_list
            ]
        )
        assert (
            shape_dynamic_track_array.shape[0]
            == shape_track_array.shape[0]
            == dynamic_track_array.shape[0]
        ), "Shape dynamic, shape and dynamic track arrays must have the same length."
        if shape_dynamic_track_array.shape[0] > 1:

            covariance_computation_shape_dynamic = compute_raw_matrix(
                shape_dynamic_track_array, t_delta=t_delta
            )

            covaraince_computation_shape = compute_raw_matrix(
                shape_track_array, t_delta=t_delta
            )

            covaraince_computation_dynamic = compute_raw_matrix(
                dynamic_track_array, t_delta=t_delta
            )
            position_computation = compute_raw_matrix(
                position_track_array, t_delta=t_delta, take_center=True
            )

            if (
                covariance_computation_shape_dynamic is not None
                and covaraince_computation_shape is not None
                and covaraince_computation_dynamic is not None
            ):

                shape_dynamic_eigenvectors = covariance_computation_shape_dynamic
                shape_eigenvectors = covaraince_computation_shape
                dynamic_eigenvectors = covaraince_computation_dynamic
                position_vectors = position_computation
                shape_dynamic_covariance_matrix.extend(shape_dynamic_eigenvectors)
                shape_covariance_matrix.extend(shape_eigenvectors)
                dynamic_covariance_matrix.extend(dynamic_eigenvectors)
                position_matrix.extend(position_vectors)
                analysis_track_ids.append(track_id)
    if (
        len(shape_dynamic_covariance_matrix) > 0
        and len(shape_covariance_matrix) > 0
        and len(dynamic_covariance_matrix) > 0
    ):

        (
            shape_dynamic_eigenvectors_1d,
            shape_eigenvectors_1d,
            dynamic_eigenvectors_1d,
            shape_dynamic_cluster_labels,
            shape_cluster_labels,
            dynamic_cluster_labels,
            shape_dynamic_linkage_matrix,
            shape_linkage_matrix,
            dynamic_linkage_matrix,
            shape_dynamic_silhouette,
            shape_dynamic_wcss_value,
            shape_silhouette,
            shape_wcss_value,
            dynamic_silhouette,
            dynamic_wcss_value,
            cluster_distance_map_shape_dynamic,
            cluster_distance_map_shape,
            cluster_distance_map_dynamic,
            cluster_eucledian_distance_map_shape_dynamic,
            cluster_eucledian_distance_map_shape,
            cluster_eucledian_distance_map_dynamic,
            analysis_track_ids,
        ) = core_clustering(
            shape_dynamic_covariance_matrix,
            shape_covariance_matrix,
            dynamic_covariance_matrix,
            position_matrix,
            analysis_track_ids,
            metric,
            method,
            cluster_threshold_shape_dynamic,
            cluster_threshold_dynamic,
            cluster_threshold_shape,
            criterion,
            distance_vectors=distance_vectors,
        )


def unsupervised_clustering(
    full_dataframe,
    csv_file_name,
    analysis_vectors,
    cluster_threshold_shape_dynamic=3,
    cluster_threshold_dynamic=3,
    cluster_threshold_shape=3,
    metric="euclidean",
    method="ward",
    criterion="maxclust",
    distance_vectors="shape",
):

    analysis_track_ids = []
    shape_dynamic_covariance_matrix = []
    shape_covariance_matrix = []
    dynamic_covariance_matrix = []
    for track_id, (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
    ) in analysis_vectors.items():
        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )
        shape_track_array = np.array(
            [[item for item in record.values()] for record in shape_dataframe_list]
        )
        dynamic_track_array = np.array(
            [[item for item in record.values()] for record in dynamic_dataframe_list]
        )
        assert (
            shape_dynamic_track_array.shape[0]
            == shape_track_array.shape[0]
            == dynamic_track_array.shape[0]
        ), "Shape dynamic, shape and dynamic track arrays must have the same length."
        if shape_dynamic_track_array.shape[0] > 1:

            covariance_computation_shape_dynamic = compute_covariance_matrix(
                shape_dynamic_track_array
            )

            covaraince_computation_shape = compute_covariance_matrix(shape_track_array)

            covaraince_computation_dynamic = compute_covariance_matrix(
                dynamic_track_array
            )
            if (
                covariance_computation_shape_dynamic is not None
                and covaraince_computation_shape is not None
                and covaraince_computation_dynamic is not None
            ):
                (
                    shape_dynamic_covariance,
                    shape_dynamic_eigenvectors,
                ) = covariance_computation_shape_dynamic
                (shape_covariance, shape_eigenvectors) = covaraince_computation_shape
                (
                    dynamic_covaraince,
                    dynamic_eigenvectors,
                ) = covaraince_computation_dynamic
                shape_dynamic_covariance_matrix.extend(shape_dynamic_covariance)
                shape_covariance_matrix.extend(shape_covariance)
                dynamic_covariance_matrix.extend(dynamic_covaraince)
                analysis_track_ids.append(track_id)
    if (
        len(shape_dynamic_covariance_matrix) > 0
        and len(shape_covariance_matrix) > 0
        and len(dynamic_covariance_matrix) > 0
    ):
        (
            shape_dynamic_eigenvectors_1d,
            shape_eigenvectors_1d,
            dynamic_eigenvectors_1d,
            shape_dynamic_cluster_labels,
            shape_cluster_labels,
            dynamic_cluster_labels,
            shape_dynamic_linkage_matrix,
            shape_linkage_matrix,
            dynamic_linkage_matrix,
            shape_dynamic_silhouette,
            shape_dynamic_wcss_value,
            shape_silhouette,
            shape_wcss_value,
            dynamic_silhouette,
            dynamic_wcss_value,
            cluster_distance_map_shape_dynamic,
            cluster_distance_map_shape,
            cluster_distance_map_dynamic,
            cluster_eucledian_distance_map_shape_dynamic,
            cluster_eucledian_distance_map_shape,
            cluster_eucledian_distance_map_dynamic,
            analysis_track_ids,
        ) = core_clustering(
            shape_dynamic_covariance_matrix,
            shape_covariance_matrix,
            dynamic_covariance_matrix,
            analysis_track_ids,
            metric,
            method,
            cluster_threshold_shape_dynamic,
            cluster_threshold_dynamic,
            cluster_threshold_shape,
            criterion,
            distance_vectors=distance_vectors,
        )


def convert_tracks_to_arrays(
    analysis_vectors,
    min_length=None,
):

    analysis_track_ids = []
    shape_dynamic_eigenvectors_matrix = []
    shape_eigenvectors_matrix = []
    dynamic_eigenvectors_matrix = []
    for track_id, (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
    ) in analysis_vectors.items():
        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )
        shape_track_array = np.array(
            [[item for item in record.values()] for record in shape_dataframe_list]
        )
        dynamic_track_array = np.array(
            [[item for item in record.values()] for record in dynamic_dataframe_list]
        )

        if (
            shape_dynamic_track_array.shape[0] > 1
            and shape_dynamic_track_array.shape[0] >= min_length
            if min_length is not None
            else True
        ):

            covariance_shape_dynamic = compute_covariance_matrix(
                shape_dynamic_track_array
            )

            covariance_shape = compute_covariance_matrix(shape_track_array)

            covariance_dynamic = compute_covariance_matrix(dynamic_track_array)
            if (
                covariance_shape_dynamic is not None
                and covariance_shape is not None
                and covariance_dynamic is not None
            ):
                (
                    shape_dynamic_covariance,
                    shape_dynamic_eigenvectors,
                ) = covariance_shape_dynamic
                shape_covariance, shape_eigenvectors = covariance_shape
                dynamic_covaraince, dynamic_eigenvectors = covariance_dynamic
                shape_dynamic_eigenvectors_matrix.extend(shape_dynamic_covariance)
                shape_eigenvectors_matrix.extend(shape_covariance)
                dynamic_eigenvectors_matrix.extend(dynamic_covaraince)
                analysis_track_ids.append(track_id)
    if (
        len(shape_dynamic_eigenvectors_matrix) > 0
        and len(dynamic_eigenvectors_matrix) > 0
        and len(shape_eigenvectors_matrix) > 0
    ):

        shape_dynamic_eigenvectors_3d = np.dstack(shape_dynamic_eigenvectors_matrix)
        shape_eigenvectors_3d = np.dstack(shape_eigenvectors_matrix)
        dynamic_eigenvectors_3d = np.dstack(dynamic_eigenvectors_matrix)

        shape_dynamic_eigenvectors_2d = shape_dynamic_eigenvectors_3d.reshape(
            len(analysis_track_ids), -1
        )
        shape_eigenvectors_2d = shape_eigenvectors_3d.reshape(
            len(analysis_track_ids), -1
        )
        dynamic_eigenvectors_2d = dynamic_eigenvectors_3d.reshape(
            len(analysis_track_ids), -1
        )

        shape_dynamic_covariance_2d = np.array(shape_dynamic_eigenvectors_2d)
        shape_covariance_2d = np.array(shape_eigenvectors_2d)
        dynamic_covariance_2d = np.array(dynamic_eigenvectors_2d)

        return (
            shape_dynamic_covariance_2d,
            shape_covariance_2d,
            dynamic_covariance_2d,
            analysis_track_ids,
        )


def cell_fate_recipe(track_data):

    dividing_shape_dynamic_dataframe = track_data[SHAPE_DYNAMIC_FEATURES].copy()

    dividing_shape_dataframe = track_data[SHAPE_FEATURES].copy()

    dividing_dynamic_dataframe = track_data[DYNAMIC_FEATURES].copy()

    dividing_shape_dynamic_dataframe.dropna(inplace=True)
    dividing_shape_dataframe.dropna(inplace=True)
    dividing_dynamic_dataframe.dropna(inplace=True)

    dividing_shape_dynamic_dataframe_list = dividing_shape_dynamic_dataframe.to_dict(
        orient="records"
    )
    dividing_shape_dataframe_list = dividing_shape_dataframe.to_dict(orient="records")
    dividing_dynamic_dataframe_list = dividing_dynamic_dataframe.to_dict(
        orient="records"
    )

    dividing_shape_dynamic_track_array = np.array(
        [
            [item for item in record.values()]
            for record in dividing_shape_dynamic_dataframe_list
        ]
    )
    dividing_shape_track_array = np.array(
        [[item for item in record.values()] for record in dividing_shape_dataframe_list]
    )
    dividing_dynamic_track_array = np.array(
        [
            [item for item in record.values()]
            for record in dividing_dynamic_dataframe_list
        ]
    )
    if dividing_shape_dynamic_track_array.shape[0] > 1:
        (
            dividing_covariance_shape_dynamic,
            dividing_eigenvectors_shape_dynamic,
        ) = compute_covariance_matrix(dividing_shape_dynamic_track_array)

        (
            dividing_covariance_shape,
            dividing_eigenvectors_shape,
        ) = compute_covariance_matrix(dividing_shape_track_array)

        (
            dividing_covariance_dynamic,
            dividing_eigenvectors_dynamic,
        ) = compute_covariance_matrix(dividing_dynamic_track_array)

        dividing_shape_dynamic_eigenvectors_3d = np.dstack(
            dividing_covariance_shape_dynamic
        )
        dividing_shape_eigenvectors_3d = np.dstack(dividing_covariance_shape)
        dividing_dynamic_eigenvectors_3d = np.dstack(dividing_covariance_dynamic)
        dividing_shape_dynamic_eigenvectors_2d = (
            dividing_shape_dynamic_eigenvectors_3d.reshape(1, -1)
        )
        dividing_shape_eigenvectors_2d = dividing_shape_eigenvectors_3d.reshape(1, -1)
        dividing_dynamic_eigenvectors_2d = dividing_dynamic_eigenvectors_3d.reshape(
            1, -1
        )

        dividing_shape_dynamic_covariance_2d = np.array(
            dividing_shape_dynamic_eigenvectors_2d
        )
        dividing_shape_covariance_2d = np.array(dividing_shape_eigenvectors_2d)
        dividing_dynamic_covariance_2d = np.array(dividing_dynamic_eigenvectors_2d)

        return (
            dividing_shape_dynamic_track_array,
            dividing_shape_track_array,
            dividing_dynamic_track_array,
            dividing_covariance_shape_dynamic,
            dividing_covariance_shape,
            dividing_covariance_dynamic,
            dividing_shape_dynamic_covariance_2d,
            dividing_shape_covariance_2d,
            dividing_dynamic_covariance_2d,
        )


def local_track_covaraince(analysis_vectors):

    analysis_track_ids = []
    shape_dynamic_eigenvectors_matrix = []
    shape_eigenvectors_matrix = []
    dynamic_eigenvectors_matrix = []
    for track_id, (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
    ) in analysis_vectors.items():
        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )
        shape_track_array = np.array(
            [[item for item in record.values()] for record in shape_dataframe_list]
        )
        dynamic_track_array = np.array(
            [[item for item in record.values()] for record in dynamic_dataframe_list]
        )

        if (
            shape_dynamic_track_array.shape[0] > 1
            and len(shape_dynamic_track_array.shape) > 1
        ):

            (
                covariance_shape_dynamic,
                eigenvectors_shape_dynamic,
            ) = compute_covariance_matrix(shape_dynamic_track_array)

            covariance_shape, eigenvectors_shape = compute_covariance_matrix(
                shape_track_array
            )

            covariance_dynamic, eigenvectors_dynamic = compute_covariance_matrix(
                dynamic_track_array
            )

            if (
                covariance_shape_dynamic is not None
                and covariance_shape is not None
                and covariance_dynamic is not None
            ):

                shape_dynamic_eigenvectors_matrix.append(covariance_shape_dynamic)
                shape_eigenvectors_matrix.append(covariance_shape)
                dynamic_eigenvectors_matrix.append(covariance_dynamic)
                analysis_track_ids.append(track_id)
    if (
        len(shape_dynamic_eigenvectors_matrix) > 0
        and len(dynamic_eigenvectors_matrix) > 0
        and len(shape_eigenvectors_matrix) > 0
    ):

        mean_shape_dynamic_eigenvectors = np.mean(
            shape_dynamic_eigenvectors_matrix, axis=0
        )
        mean_shape_eigenvectors = np.mean(shape_eigenvectors_matrix, axis=0)
        mean_dynamic_eigenvectors = np.mean(dynamic_eigenvectors_matrix, axis=0)

        return (
            mean_shape_dynamic_eigenvectors,
            mean_shape_eigenvectors,
            mean_dynamic_eigenvectors,
            analysis_track_ids,
        )


def convert_tracks_to_simple_arrays(
    analysis_vectors,
    metric="euclidean",
    cluster_threshold_shape_dynamic=4,
    cluster_threshold_dynamic=4,
    cluster_threshold_shape=4,
    method="ward",
    criterion="maxclust",
    t_delta=10,
    distance_vectors="shape",
):

    analysis_track_ids = []
    shape_dynamic_eigenvectors_matrix = []
    shape_eigenvectors_matrix = []
    dynamic_eigenvectors_matrix = []
    position_matrix = []
    for track_id, (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
    ) in analysis_vectors.items():
        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )
        shape_track_array = np.array(
            [[item for item in record.values()] for record in shape_dataframe_list]
        )
        dynamic_track_array = np.array(
            [[item for item in record.values()] for record in dynamic_dataframe_list]
        )
        columns_of_interest = ["z", "y", "x"]
        position_track_array = np.array(
            [
                [record[col] for col in columns_of_interest]
                for record in full_dataframe_list
            ]
        )

        if (
            shape_dynamic_track_array.shape[0] > 1
            and len(shape_dynamic_track_array.shape) > 1
        ):

            covariance_shape_dynamic = compute_raw_matrix(
                shape_dynamic_track_array, t_delta=t_delta
            )
            covariance_shape = compute_raw_matrix(shape_track_array, t_delta=t_delta)

            covariance_dynamic = compute_raw_matrix(
                dynamic_track_array, t_delta=t_delta
            )

            position_computation = compute_raw_matrix(
                position_track_array, t_delta=t_delta, take_center=True
            )

            if (
                covariance_shape_dynamic is not None
                and covariance_shape is not None
                and covariance_dynamic is not None
            ):

                shape_dynamic_eigenvectors = covariance_shape_dynamic
                shape_eigenvectors = covariance_shape
                dynamic_eigenvectors = covariance_dynamic
                position_vectors = position_computation
                shape_dynamic_eigenvectors_matrix.extend(shape_dynamic_eigenvectors)
                shape_eigenvectors_matrix.extend(shape_eigenvectors)
                dynamic_eigenvectors_matrix.extend(dynamic_eigenvectors)
                position_matrix.extend(position_vectors)
                analysis_track_ids.append(track_id)
    if (
        len(shape_dynamic_eigenvectors_matrix) > 0
        and len(dynamic_eigenvectors_matrix) > 0
        and len(shape_eigenvectors_matrix) > 0
    ):

        (
            shape_dynamic_eigenvectors_1d,
            shape_eigenvectors_1d,
            dynamic_eigenvectors_1d,
            shape_dynamic_cluster_labels,
            shape_cluster_labels,
            dynamic_cluster_labels,
            shape_dynamic_linkage_matrix,
            shape_linkage_matrix,
            dynamic_linkage_matrix,
            shape_dynamic_silhouette,
            shape_dynamic_wcss_value,
            shape_silhouette,
            shape_wcss_value,
            dynamic_silhouette,
            dynamic_wcss_value,
            cluster_distance_map_shape_dynamic,
            cluster_distance_map_shape,
            cluster_distance_map_dynamic,
            cluster_eucledian_distance_map_shape_dynamic,
            cluster_eucledian_distance_map_shape,
            cluster_eucledian_distance_map_dynamic,
            analysis_track_ids,
        ) = core_clustering(
            shape_dynamic_eigenvectors_matrix,
            shape_eigenvectors_matrix,
            dynamic_eigenvectors_matrix,
            position_matrix,
            analysis_track_ids,
            metric,
            method,
            cluster_threshold_shape_dynamic,
            cluster_threshold_dynamic,
            cluster_threshold_shape,
            criterion,
            distance_vectors=distance_vectors,
        )

        shape_dynamic_cluster_labels_dict = {
            track_id: cluster_label
            for track_id, cluster_label in zip(
                analysis_track_ids, shape_dynamic_cluster_labels
            )
        }
        shape_cluster_labels_dict = {
            track_id: cluster_label
            for track_id, cluster_label in zip(analysis_track_ids, shape_cluster_labels)
        }
        dynamic_cluster_labels_dict = {
            track_id: cluster_label
            for track_id, cluster_label in zip(
                analysis_track_ids, dynamic_cluster_labels
            )
        }

        cluster_distance_map_shape_dynamic_dict = {
            track_id: cluster_distance_map_shape_dynamic[cluster_label]
            for track_id, cluster_label in zip(
                analysis_track_ids, shape_dynamic_cluster_labels
            )
        }

        cluster_distance_map_shape_dict = {
            track_id: cluster_distance_map_shape[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, shape_cluster_labels)
        }

        cluster_distance_map_dynamic_dict = {
            track_id: cluster_distance_map_dynamic[cluster_label]
            for track_id, cluster_label in zip(
                analysis_track_ids, dynamic_cluster_labels
            )
        }

        cluster_eucledian_distance_map_shape_dynamic_dict = {
            track_id: cluster_eucledian_distance_map_shape_dynamic[cluster_label]
            for track_id, cluster_label in zip(
                analysis_track_ids, shape_dynamic_cluster_labels
            )
        }

        cluster_eucledian_distance_map_shape_dict = {
            track_id: cluster_eucledian_distance_map_shape[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, shape_cluster_labels)
        }

        cluster_eucledian_distance_map_dynamic_dict = {
            track_id: cluster_eucledian_distance_map_dynamic[cluster_label]
            for track_id, cluster_label in zip(
                analysis_track_ids, dynamic_cluster_labels
            )
        }

        return (
            shape_dynamic_eigenvectors_1d,
            shape_eigenvectors_1d,
            dynamic_eigenvectors_1d,
            shape_dynamic_cluster_labels_dict,
            shape_cluster_labels_dict,
            dynamic_cluster_labels_dict,
            shape_dynamic_linkage_matrix,
            shape_linkage_matrix,
            dynamic_linkage_matrix,
            shape_dynamic_silhouette,
            shape_dynamic_wcss_value,
            shape_silhouette,
            shape_wcss_value,
            dynamic_silhouette,
            dynamic_wcss_value,
            cluster_distance_map_shape_dynamic_dict,
            cluster_distance_map_shape_dict,
            cluster_distance_map_dynamic_dict,
            cluster_eucledian_distance_map_shape_dynamic_dict,
            cluster_eucledian_distance_map_shape_dict,
            cluster_eucledian_distance_map_dynamic_dict,
            analysis_track_ids,
        )


def convert_pseudo_tracks_to_simple_arrays(
    analysis_vectors,
    t_delta=10,
    distance_vectors="shape",
):

    analysis_track_ids = []
    shape_dynamic_eigenvectors_matrix = []
    shape_eigenvectors_matrix = []
    dynamic_eigenvectors_matrix = []
    position_matrix = []
    cell_type_ids = []
    for track_id, (
        shape_dynamic_dataframe_list,
        shape_dataframe_list,
        dynamic_dataframe_list,
        full_dataframe_list,
        cell_type_label,
    ) in analysis_vectors.items():
        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )
        shape_track_array = np.array(
            [[item for item in record.values()] for record in shape_dataframe_list]
        )
        dynamic_track_array = np.array(
            [[item for item in record.values()] for record in dynamic_dataframe_list]
        )
        columns_of_interest = ["z", "y", "x"]
        position_track_array = np.array(
            [
                [record[col] for col in columns_of_interest]
                for record in full_dataframe_list
            ]
        )

        if (
            shape_dynamic_track_array.shape[0] > 1
            and len(shape_dynamic_track_array.shape) > 1
        ):

            covariance_shape_dynamic = compute_raw_matrix(
                shape_dynamic_track_array, t_delta=t_delta
            )
            covariance_shape = compute_raw_matrix(shape_track_array, t_delta=t_delta)

            covariance_dynamic = compute_raw_matrix(
                dynamic_track_array, t_delta=t_delta
            )

            position_computation = compute_raw_matrix(
                position_track_array, t_delta=t_delta, take_center=True
            )

            if (
                covariance_shape_dynamic is not None
                and covariance_shape is not None
                and covariance_dynamic is not None
            ):

                shape_dynamic_eigenvectors = covariance_shape_dynamic
                shape_eigenvectors = covariance_shape
                dynamic_eigenvectors = covariance_dynamic
                position_vectors = position_computation
                shape_dynamic_eigenvectors_matrix.extend(shape_dynamic_eigenvectors)
                shape_eigenvectors_matrix.extend(shape_eigenvectors)
                dynamic_eigenvectors_matrix.extend(dynamic_eigenvectors)
                position_matrix.extend(position_vectors)
                analysis_track_ids.append(track_id)
                cell_type_ids.append(cell_type_label)
    if (
        len(shape_dynamic_eigenvectors_matrix) > 0
        and len(dynamic_eigenvectors_matrix) > 0
        and len(shape_eigenvectors_matrix) > 0
    ):

        (
            shape_dynamic_eigenvectors_1d,
            shape_eigenvectors_1d,
            dynamic_eigenvectors_1d,
            cluster_distance_map_shape_dynamic,
            cluster_eucledian_distance_map_shape_dynamic,
            cluster_distance_map_dynamic,
            cluster_eucledian_distance_map_dynamic,
            cluster_distance_map_shape,
            cluster_eucledian_distance_map_shape,
            analysis_track_ids,
            cell_type_ids,
        ) = pseudo_core_clustering(
            shape_dynamic_eigenvectors_matrix,
            shape_eigenvectors_matrix,
            dynamic_eigenvectors_matrix,
            position_matrix,
            analysis_track_ids,
            cell_type_ids,
            distance_vectors=distance_vectors,
        )

        shape_dynamic_cluster_labels_dict = {
            track_id: cluster_label
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }
        shape_cluster_labels_dict = {
            track_id: cluster_label
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }
        dynamic_cluster_labels_dict = {
            track_id: cluster_label
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        cluster_distance_map_shape_dynamic_dict = {
            track_id: cluster_distance_map_shape_dynamic[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        cluster_distance_map_shape_dict = {
            track_id: cluster_distance_map_shape[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        cluster_distance_map_dynamic_dict = {
            track_id: cluster_distance_map_dynamic[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        cluster_eucledian_distance_map_shape_dynamic_dict = {
            track_id: cluster_eucledian_distance_map_shape_dynamic[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        cluster_eucledian_distance_map_shape_dict = {
            track_id: cluster_eucledian_distance_map_shape[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        cluster_eucledian_distance_map_dynamic_dict = {
            track_id: cluster_eucledian_distance_map_dynamic[cluster_label]
            for track_id, cluster_label in zip(analysis_track_ids, cell_type_ids)
        }

        return (
            shape_dynamic_eigenvectors_1d,
            shape_eigenvectors_1d,
            dynamic_eigenvectors_1d,
            shape_dynamic_cluster_labels_dict,
            shape_cluster_labels_dict,
            dynamic_cluster_labels_dict,
            cluster_distance_map_shape_dynamic_dict,
            cluster_distance_map_shape_dict,
            cluster_distance_map_dynamic_dict,
            cluster_eucledian_distance_map_shape_dynamic_dict,
            cluster_eucledian_distance_map_shape_dict,
            cluster_eucledian_distance_map_dynamic_dict,
            analysis_track_ids,
        )


def core_clustering(
    shape_dynamic_eigenvectors_matrix,
    shape_eigenvectors_matrix,
    dynamic_eigenvectors_matrix,
    position_matrix,
    analysis_track_ids,
    metric,
    method,
    cluster_threshold_shape_dynamic_range,
    cluster_threshold_dynamic_range,
    cluster_threshold_shape_range,
    criterion,
    distance_vectors="shape",
):

    best_threshold_shape_dynamic = None
    best_silhouette_shape_dynamic = -np.inf
    best_wcss_shape_dynamic_value = np.inf
    best_threshold_dynamic = None
    best_silhouette_dynamic = -np.inf
    best_wcss_dynamic_value = np.inf
    best_threshold_shape = None
    best_silhouette_shape = -np.inf
    best_wcss_shape_value = np.inf
    best_shape_dynamic_cluster_labels = None
    best_dynamic_cluster_labels = None
    best_shape_cluster_labels = None
    best_shape_dynamic_linkage_matrix = None
    best_dynamic_linkage_matrix = None
    best_shape_linkage_matrix = None
    best_shape_dynamic_linkage_matrix = None

    best_cluster_distance_map_shape_dynamic = None
    best_cluster_distance_map_dynamic = None
    best_cluster_distance_map_shape = None
    best_cluster_eucledian_distance_map_shape_dynamic = None
    best_cluster_eucledian_distance_map_dynamic = None
    best_cluster_eucledian_distance_map_shape = None

    if isinstance(cluster_threshold_shape_dynamic_range, int):
        cluster_threshold_shape_dynamic_range = [cluster_threshold_shape_dynamic_range]
    if isinstance(cluster_threshold_dynamic_range, int):
        cluster_threshold_dynamic_range = [cluster_threshold_dynamic_range]
    if isinstance(cluster_threshold_shape_range, int):
        cluster_threshold_shape_range = [cluster_threshold_shape_range]
    shape_dynamic_eigenvectors_3d = np.dstack(shape_dynamic_eigenvectors_matrix)
    shape_eigenvectors_3d = np.dstack(shape_eigenvectors_matrix)
    dynamic_eigenvectors_3d = np.dstack(dynamic_eigenvectors_matrix)
    position_vectors_3d = np.dstack(position_matrix)

    position_vector_2d = position_vectors_3d.reshape(len(analysis_track_ids), -1)
    shape_dynamic_eigenvectors_2d = shape_dynamic_eigenvectors_3d.reshape(
        len(analysis_track_ids), -1
    )
    shape_eigenvectors_2d = shape_eigenvectors_3d.reshape(len(analysis_track_ids), -1)
    dynamic_eigenvectors_2d = dynamic_eigenvectors_3d.reshape(
        len(analysis_track_ids), -1
    )

    shape_dynamic_eigenvectors_1d = np.array(shape_dynamic_eigenvectors_2d)
    shape_eigenvectors_1d = np.array(shape_eigenvectors_2d)
    dynamic_eigenvectors_1d = np.array(dynamic_eigenvectors_2d)
    position_vector_1d = np.array(position_vector_2d)
    compute_vectors_shape = shape_eigenvectors_1d
    compute_vectors_dynamic = dynamic_eigenvectors_1d
    if distance_vectors == "shape":
        compute_vectors = shape_eigenvectors_1d
    if distance_vectors == "dynamic":
        compute_vectors = dynamic_eigenvectors_1d
    if distance_vectors == "shape_and_dynamic":
        compute_vectors = shape_dynamic_eigenvectors_1d
    else:
        compute_vectors = shape_eigenvectors_1d

    for cluster_threshold_shape_dynamic in cluster_threshold_shape_dynamic_range:

        shape_dynamic_cosine_distance = pdist(
            shape_dynamic_eigenvectors_1d, metric=metric
        )

        shape_dynamic_linkage_matrix = linkage(
            shape_dynamic_cosine_distance, method=method
        )
        try:
            shape_dynamic_cluster_labels = fcluster(
                shape_dynamic_linkage_matrix,
                cluster_threshold_shape_dynamic,
                criterion=criterion,
            )
        except Exception:
            shape_dynamic_cluster_labels = fcluster(
                shape_dynamic_linkage_matrix, 1, criterion="maxclust"
            )

        cluster_distance_map_shape_dynamic = calculate_intercluster_distance(
            compute_vectors, shape_dynamic_cluster_labels
        )
        cluster_eucledian_distance_map_shape_dynamic = (
            calculate_intercluster_eucledian_distance(
                position_vector_1d, shape_dynamic_cluster_labels
            )
        )

        shape_dynamic_cluster_centroids = calculate_cluster_centroids(
            shape_dynamic_eigenvectors_1d, shape_dynamic_cluster_labels
        )
        try:
            shape_dynamic_silhouette = silhouette_score(
                shape_dynamic_eigenvectors_1d,
                shape_dynamic_cluster_labels,
                metric=metric,
            )
        except Exception:
            shape_dynamic_silhouette = 0
        try:
            shape_dynamic_wcss_value = calculate_wcss(
                shape_dynamic_eigenvectors_1d,
                shape_dynamic_cluster_labels,
                shape_dynamic_cluster_centroids,
            )
        except Exception:
            shape_dynamic_wcss_value = np.inf

        sil_condition = (shape_dynamic_silhouette > best_silhouette_shape_dynamic) & (
            shape_dynamic_silhouette > 0
        )

        condition = sil_condition
        if condition:
            best_silhouette_shape_dynamic = shape_dynamic_silhouette
            best_threshold_shape_dynamic = cluster_threshold_shape_dynamic
            best_wcss_shape_dynamic_value = shape_dynamic_wcss_value
            best_shape_dynamic_cluster_labels = shape_dynamic_cluster_labels
            best_shape_dynamic_linkage_matrix = shape_dynamic_linkage_matrix
            best_cluster_distance_map_shape_dynamic = cluster_distance_map_shape_dynamic
            best_cluster_eucledian_distance_map_shape_dynamic = (
                cluster_eucledian_distance_map_shape_dynamic
            )

    for cluster_threshold_dynamic in cluster_threshold_dynamic_range:

        dynamic_cosine_distance = pdist(dynamic_eigenvectors_1d, metric=metric)

        dynamic_linkage_matrix = linkage(dynamic_cosine_distance, method=method)
        try:
            dynamic_cluster_labels = fcluster(
                dynamic_linkage_matrix,
                cluster_threshold_dynamic,
                criterion=criterion,
            )
        except Exception:
            dynamic_cluster_labels = fcluster(
                dynamic_linkage_matrix,
                1,
                criterion="maxclust",
            )

        cluster_distance_map_dynamic = calculate_intercluster_distance(
            compute_vectors_shape, dynamic_cluster_labels
        )
        cluster_eucledian_distance_map_dynamic = (
            calculate_intercluster_eucledian_distance(
                position_vector_1d, dynamic_cluster_labels
            )
        )

        dynamic_cluster_centroids = calculate_cluster_centroids(
            dynamic_eigenvectors_1d, dynamic_cluster_labels
        )
        try:
            dynamic_silhouette = silhouette_score(
                dynamic_eigenvectors_1d, dynamic_cluster_labels, metric=metric
            )
        except Exception:
            dynamic_silhouette = 0
        try:
            dynamic_wcss_value = calculate_wcss(
                dynamic_eigenvectors_1d,
                dynamic_cluster_labels,
                dynamic_cluster_centroids,
            )
        except Exception:
            dynamic_wcss_value = np.inf
        sil_condition = (dynamic_silhouette > best_silhouette_dynamic) & (
            dynamic_silhouette > 0
        )

        condition = sil_condition
        if condition:
            best_silhouette_dynamic = dynamic_silhouette
            best_threshold_dynamic = cluster_threshold_dynamic
            best_wcss_dynamic_value = dynamic_wcss_value
            best_dynamic_cluster_labels = dynamic_cluster_labels
            best_dynamic_linkage_matrix = dynamic_linkage_matrix
            best_cluster_distance_map_dynamic = cluster_distance_map_dynamic
            best_cluster_eucledian_distance_map_dynamic = (
                cluster_eucledian_distance_map_dynamic
            )

    for cluster_threshold_shape in cluster_threshold_shape_range:
        shape_cosine_distance = pdist(shape_eigenvectors_1d, metric=metric)

        shape_linkage_matrix = linkage(shape_cosine_distance, method=method)
        try:
            shape_cluster_labels = fcluster(
                shape_linkage_matrix, cluster_threshold_shape, criterion=criterion
            )
        except Exception:
            shape_cluster_labels = fcluster(
                shape_linkage_matrix, 1, criterion="maxclust"
            )

        shape_cluster_centroids = calculate_cluster_centroids(
            shape_eigenvectors_1d, shape_cluster_labels
        )
        cluster_distance_map_shape = calculate_intercluster_distance(
            compute_vectors_dynamic, shape_cluster_labels
        )
        cluster_eucledian_distance_map_shape = (
            calculate_intercluster_eucledian_distance(
                position_vector_1d, shape_cluster_labels
            )
        )

        try:
            shape_silhouette = silhouette_score(
                shape_eigenvectors_1d, shape_cluster_labels, metric=metric
            )
        except Exception:
            shape_silhouette = 0
        try:
            shape_wcss_value = calculate_wcss(
                shape_eigenvectors_1d, shape_cluster_labels, shape_cluster_centroids
            )
        except Exception:
            shape_wcss_value = np.inf
        sil_condition = (shape_silhouette > best_silhouette_shape) & (
            shape_silhouette > 0
        )

        condition = sil_condition
        if condition:
            best_silhouette_shape = shape_silhouette
            best_threshold_shape = cluster_threshold_shape
            best_wcss_shape_value = shape_wcss_value
            best_shape_cluster_labels = shape_cluster_labels
            best_shape_linkage_matrix = shape_linkage_matrix
            best_cluster_distance_map_shape = cluster_distance_map_shape
            best_cluster_eucledian_distance_map_shape = (
                cluster_eucledian_distance_map_shape
            )

    print(
        f"best threshold value for shape dynamic {best_threshold_shape_dynamic} with silhouette score of {best_silhouette_shape_dynamic} and with wcss score of {best_wcss_shape_dynamic_value}, number of clusters {len(np.unique(best_shape_dynamic_cluster_labels))}, total track ids {len(analysis_track_ids)}"
    )

    print(
        f"best threshold value for dynamic {best_threshold_dynamic} with silhouette score of {best_silhouette_dynamic} and with wcss score of {best_wcss_dynamic_value}, number of clusters {len(np.unique(best_dynamic_cluster_labels))}, total track ids {len(analysis_track_ids)}"
    )

    print(
        f"best threshold value for shape {best_threshold_shape} with silhouette score of {best_silhouette_shape} and with wcss score of {best_wcss_shape_value}, number of clusters {len(np.unique(best_shape_cluster_labels))}, total track ids {len(analysis_track_ids)}"
    )

    return (
        shape_dynamic_eigenvectors_1d,
        shape_eigenvectors_1d,
        dynamic_eigenvectors_1d,
        best_shape_dynamic_cluster_labels,
        best_shape_cluster_labels,
        best_dynamic_cluster_labels,
        best_shape_dynamic_linkage_matrix,
        best_shape_linkage_matrix,
        best_dynamic_linkage_matrix,
        best_silhouette_shape_dynamic,
        best_wcss_shape_dynamic_value,
        best_silhouette_shape,
        best_wcss_shape_value,
        best_silhouette_dynamic,
        best_wcss_dynamic_value,
        best_cluster_distance_map_shape_dynamic,
        best_cluster_distance_map_shape,
        best_cluster_distance_map_dynamic,
        best_cluster_eucledian_distance_map_shape_dynamic,
        best_cluster_eucledian_distance_map_shape,
        best_cluster_eucledian_distance_map_dynamic,
        analysis_track_ids,
    )


def pseudo_core_clustering(
    shape_dynamic_eigenvectors_matrix,
    shape_eigenvectors_matrix,
    dynamic_eigenvectors_matrix,
    position_matrix,
    analysis_track_ids,
    cell_type_ids,
    distance_vectors="shape",
):

    shape_dynamic_eigenvectors_3d = np.dstack(shape_dynamic_eigenvectors_matrix)
    shape_eigenvectors_3d = np.dstack(shape_eigenvectors_matrix)
    dynamic_eigenvectors_3d = np.dstack(dynamic_eigenvectors_matrix)
    position_vectors_3d = np.dstack(position_matrix)

    position_vector_2d = position_vectors_3d.reshape(len(analysis_track_ids), -1)
    shape_dynamic_eigenvectors_2d = shape_dynamic_eigenvectors_3d.reshape(
        len(analysis_track_ids), -1
    )
    shape_eigenvectors_2d = shape_eigenvectors_3d.reshape(len(analysis_track_ids), -1)
    dynamic_eigenvectors_2d = dynamic_eigenvectors_3d.reshape(
        len(analysis_track_ids), -1
    )

    shape_dynamic_eigenvectors_1d = np.array(shape_dynamic_eigenvectors_2d)
    shape_eigenvectors_1d = np.array(shape_eigenvectors_2d)
    dynamic_eigenvectors_1d = np.array(dynamic_eigenvectors_2d)
    position_vector_1d = np.array(position_vector_2d)
    compute_vectors_shape = shape_eigenvectors_1d
    compute_vectors_dynamic = dynamic_eigenvectors_1d
    if distance_vectors == "shape":
        compute_vectors = shape_eigenvectors_1d
    if distance_vectors == "dynamic":
        compute_vectors = dynamic_eigenvectors_1d
    if distance_vectors == "shape_and_dynamic":
        compute_vectors = shape_dynamic_eigenvectors_1d
    else:
        compute_vectors = shape_eigenvectors_1d

    cluster_distance_map_shape_dynamic = calculate_intercluster_distance(
        compute_vectors, cell_type_ids
    )
    cluster_eucledian_distance_map_shape_dynamic = (
        calculate_intercluster_eucledian_distance(position_vector_1d, cell_type_ids)
    )

    cluster_distance_map_dynamic = calculate_intercluster_distance(
        compute_vectors_shape, cell_type_ids
    )
    cluster_eucledian_distance_map_dynamic = calculate_intercluster_eucledian_distance(
        position_vector_1d, cell_type_ids
    )

    cluster_distance_map_shape = calculate_intercluster_distance(
        compute_vectors_dynamic, cell_type_ids
    )
    cluster_eucledian_distance_map_shape = calculate_intercluster_eucledian_distance(
        position_vector_1d, cell_type_ids
    )

    return (
        shape_dynamic_eigenvectors_1d,
        shape_eigenvectors_1d,
        dynamic_eigenvectors_1d,
        cluster_distance_map_shape_dynamic,
        cluster_eucledian_distance_map_shape_dynamic,
        cluster_distance_map_dynamic,
        cluster_eucledian_distance_map_dynamic,
        cluster_distance_map_shape,
        cluster_eucledian_distance_map_shape,
        analysis_track_ids,
        cell_type_ids,
    )


def compute_raw_matrix(track_arrays, t_delta, take_center=False):
    track_duration = track_arrays.shape[0]
    t_delta = int(t_delta)

    if track_duration < t_delta:
        repetitions = t_delta - track_duration
        last_row = track_arrays[-1, :]
        repeated_rows = np.tile(last_row, (repetitions, 1))
        result_matrix = np.vstack([track_arrays, repeated_rows])
    elif track_duration > t_delta:
        result_matrix = track_arrays[:t_delta, :]
    else:
        result_matrix = track_arrays
    if take_center:
        flattened_array = result_matrix[result_matrix.shape[0] // 2, :]
    else:
        flattened_array = result_matrix.flatten()

    return flattened_array


def compute_covariance_matrix(track_arrays):
    try:
        covariance_matrix = np.cov(track_arrays, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        eigenvalue_order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalue_order]
        eigenvectors = eigenvectors[:, eigenvalue_order]
        normalized_eigenvectors = np.array(
            [v / np.linalg.norm(v) for v in eigenvectors]
        )

        real_part = np.real(normalized_eigenvectors)
        imag_part = np.imag(normalized_eigenvectors)

        concatenated_eigenvectors = np.concatenate((real_part, imag_part), axis=1)

        return covariance_matrix, concatenated_eigenvectors
    except Exception as e:
        print(f"Covariance matric computation {e}")


def train_gbr_neural_net(
    save_path,
    h5_file=None,
    num_classes=2,
    batch_size=64,
    num_workers=0,
    learning_rate=0.001,
    epochs=100,
    accelerator="cuda",
    devices=1,
    model_type="attention",
    growth_rate: int = 32,
    block_config: tuple = (6, 12, 24, 16),
    num_init_features: int = 32,
    bottleneck_size: int = 4,
    kernel_size: int = 3,
    experiment_name="mitosis",
    scheduler_choice="plateau",
    attention_dim: int = 64,
    n_pos: list = (8,),
    mean=0.0,
    std=0.02,
    min_scale=0.95,
    max_shift=1.05,
    max_scale=1.05,
    max_mask_ratio=0.1,
    augment = False
):

    if isinstance(block_config, int):
        block_config = (block_config,)
    mitosis_inception = MitosisInception(
        h5_file=h5_file,
        num_classes=num_classes,
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features,
        bottleneck_size=bottleneck_size,
        kernel_size=kernel_size,
        num_workers=num_workers,
        epochs=epochs,
        log_path=save_path,
        batch_size=batch_size,
        accelerator=accelerator,
        devices=devices,
        experiment_name=experiment_name,
        scheduler_choice=scheduler_choice,
        learning_rate=learning_rate,
        n_pos=n_pos,
        attention_dim=attention_dim,
    )
    
    if augment:
            mitosis_inception.setup_timeseries_transforms(
                mean=mean,
                std=std,
                min_scale=min_scale,
                max_shift=max_shift,
                max_scale=max_scale,
                max_mask_ratio=max_mask_ratio,
            )
    else:
        mitosis_inception.time_series_transforms = None        
        
    mitosis_inception.setup_gbr_h5_datasets()

    if model_type == "simple":
        mitosis_inception.setup_mitosisnet_model()
    if model_type == "densenet":
        mitosis_inception.setup_densenet_model()
    if model_type == "attention":
        mitosis_inception.setup_hybrid_attention_model()

    mitosis_inception.setup_logger()
    mitosis_inception.setup_checkpoint()
    mitosis_inception.setup_adam()
    mitosis_inception.setup_lightning_model()
    mitosis_inception.train()


def train_mitosis_neural_net(
    save_path,
    h5_file=None,
    num_classes=2,
    batch_size=64,
    num_workers=0,
    learning_rate=0.001,
    epochs=100,
    accelerator="cuda",
    devices=1,
    model_type="attention",
    growth_rate: int = 32,
    block_config: tuple = (6, 12, 24, 16),
    num_init_features: int = 32,
    bottleneck_size: int = 4,
    kernel_size: int = 3,
    experiment_name="mitosis",
    scheduler_choice="plateau",
    attention_dim: int = 64,
    n_pos: list = (8,),
):

    if isinstance(block_config, int):
        block_config = (block_config,)

    mitosis_inception = MitosisInception(
        h5_file=h5_file,
        num_classes=num_classes,
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features,
        bottleneck_size=bottleneck_size,
        kernel_size=kernel_size,
        num_workers=num_workers,
        epochs=epochs,
        log_path=save_path,
        batch_size=batch_size,
        accelerator=accelerator,
        devices=devices,
        experiment_name=experiment_name,
        scheduler_choice=scheduler_choice,
        learning_rate=learning_rate,
        n_pos=n_pos,
        attention_dim=attention_dim,
    )

    mitosis_inception.setup_timeseries_transforms()
    mitosis_inception.setup_h5_datasets()
    if model_type == "simple":
        mitosis_inception.setup_mitosisnet_model()
    if model_type == "densenet":
        mitosis_inception.setup_densenet_model()
    if model_type == "attention":
        mitosis_inception.setup_hybrid_attention_model()

    mitosis_inception.setup_logger()
    mitosis_inception.setup_checkpoint()
    mitosis_inception.setup_adam()
    mitosis_inception.setup_lightning_model()
    mitosis_inception.train()


def train_gbr_vision_neural_net(
    save_path,
    h5_file,
    input_shape,
    box_vector=7,
    start_kernel=7,
    mid_kernel=3,
    startfilter=64,
    growth_rate=32,
    depth={"depth_0": 12, "depth_1": 24, "depth_2": 16},
    num_classes=3,
    batch_size=64,
    num_workers=0,
    learning_rate=0.001,
    epochs=100,
    accelerator="cuda",
    devices=1,
    loss_function="oneat",
    experiment_name="mitosis",
    scheduler_choice="plateau",
    oneat_accuracy=True,
    crop_size=None,
    pool_first=True,
):

    mitosis_inception = MitosisInception(
        h5_file=h5_file,
        num_classes=num_classes,
        num_workers=num_workers,
        epochs=epochs,
        log_path=save_path,
        batch_size=batch_size,
        accelerator=accelerator,
        devices=devices,
        experiment_name=experiment_name,
        scheduler_choice=scheduler_choice,
        loss_function=loss_function,
        learning_rate=learning_rate,
    )

    mitosis_inception.setup_gbr_vision_h5_datasets(crop_size=crop_size)

    mitosis_inception.setup_densenet_vision_model(
        input_shape,
        num_classes,
        box_vector,
        start_kernel,
        mid_kernel,
        startfilter,
        depth,
        growth_rate,
        pool_first=pool_first,
    )

    mitosis_inception.setup_logger()
    mitosis_inception.setup_checkpoint()
    mitosis_inception.setup_adam()
    mitosis_inception.setup_lightning_model(oneat_accuracy=oneat_accuracy)
    mitosis_inception.train()


def plot_metrics_from_npz(npz_file):
    data = np.load(npz_file)

    train_loss_class = data["train_loss_class1"]
    val_loss_class = data["val_loss_class1"]
    train_acc_class = data["train_acc_class1"]
    val_acc_class = data["val_acc_class1"]

    epochs = len(train_loss_class)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss_class, label="Train Loss Class 1")
    plt.plot(range(epochs), val_loss_class, label="Validation Loss Class 1")
    plt.legend(loc="upper right")
    plt.title("Loss for Class 1")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_acc_class, label="Train Acc Class 1")
    plt.plot(range(epochs), val_acc_class, label="Validation Acc Class 1")
    plt.legend(loc="upper right")
    plt.title("Accuracy for Class 1")

    plt.tight_layout()
    plt.show()


def get_zero_gen_daughter_generations(
    unique_trackmate_track_ids,
    tracks_dataframe: pd.DataFrame,
    zero_gen_tracklets,
    daughter_generations,
):

    for trackmate_track_id in unique_trackmate_track_ids:
        subset = tracks_dataframe[
            (tracks_dataframe["TrackMate Track ID"] == trackmate_track_id)
        ].sort_values(by="t")
        sorted_subset = sorted(subset["Track ID"].unique())
        for tracklet_id in sorted_subset:

            dividing_track_track_data = tracks_dataframe[
                (tracks_dataframe["Track ID"] == tracklet_id)
            ].sort_values(by="t")
            generation_id = int(
                float(dividing_track_track_data["Generation ID"].iloc[0])
            )

            track_start_time = dividing_track_track_data["t"].min()
            track_end_time = dividing_track_track_data["t"].max()

            if generation_id == 0:
                zero_gen_tracklets[trackmate_track_id] = (
                    tracklet_id,
                    track_start_time,
                    track_end_time,
                )
            elif generation_id > 0:
                if trackmate_track_id in daughter_generations[generation_id]:
                    daughter_generations[generation_id][trackmate_track_id].append(
                        (tracklet_id, track_start_time, track_end_time)
                    )
                else:
                    daughter_generations[generation_id][trackmate_track_id] = [
                        (tracklet_id, track_start_time, track_end_time)
                    ]


def populate_zero_gen_tracklets(
    zero_gen_tracklets,
    tracks_dataframe,
    zero_gen_life,
    zero_gen_polynomial_coefficients,
    zero_gen_polynomials,
    zero_gen_polynomial_time,
    zero_gen_autocorrelation,
    zero_gen_crosscorrelation,
    zero_gen_covariance,
    zero_gen_raw,
    shape_analysis=True,
    cross_analysis=False,
):

    good_zero_gens = 0
    for trackmate_track_id in zero_gen_tracklets.keys():

        zero_gen_tracklet_id, start_time, end_time = zero_gen_tracklets[
            trackmate_track_id
        ]
        if end_time - start_time > 0:
            good_zero_gens += 1
            start_end_times = np.vstack((start_time, end_time))

            zero_gen_life.append(end_time - start_time)
            dividing_track_track_data = tracks_dataframe[
                (tracks_dataframe["Track ID"] == zero_gen_tracklet_id)
            ].sort_values(by="t")
            result = cell_fate_recipe(dividing_track_track_data)
            if result is not None:
                (
                    dividing_shape_dynamic_track_array,
                    dividing_shape_track_array,
                    dividing_dynamic_track_array,
                    dividing_covariance_shape_dynamic,
                    dividing_covariance_shape,
                    dividing_covariance_dynamic,
                    dividing_shape_dynamic_prediction_input,
                    dividing_shape_prediction_input,
                    dividing_dynamic_prediction_input,
                ) = result

                if shape_analysis:

                    track_array = dividing_shape_track_array
                    feature = SHAPE_FEATURES

                elif not shape_analysis and not cross_analysis:

                    track_array = dividing_dynamic_track_array
                    feature = DYNAMIC_FEATURES

                elif cross_analysis and not shape_analysis:

                    track_array = dividing_shape_dynamic_track_array
                    feature = SHAPE_DYNAMIC_FEATURES

                generic_polynomial_fits(
                    track_array,
                    start_time,
                    start_end_times,
                    zero_gen_polynomial_coefficients,
                    zero_gen_polynomials,
                    zero_gen_polynomial_time,
                    zero_gen_autocorrelation,
                    zero_gen_crosscorrelation,
                    zero_gen_covariance,
                    zero_gen_raw,
                    feature,
                )


def generic_polynomial_fits(
    track_array,
    start_time,
    start_end_times,
    nth_generation_polynomial_coefficients,
    nth_generation_polynomials,
    nth_generation_polynomial_time,
    nth_generation_autocorrelation,
    nth_generation_crosscorrelation,
    nth_generation_covariance,
    nth_generation_raw,
    feature,
    polynomial_degree=6,
    nlags=50,
):
    matrix_t_f = np.transpose(track_array)
    cov_matrix = np.cov(matrix_t_f)
    poly_feature = PolynomialFeatures(degree=polynomial_degree, include_bias=True)

    for i in range(matrix_t_f.shape[0]):
        all_cross_correlations = []
        track_shape_feature = matrix_t_f[i]
        x = np.arange(start_time, matrix_t_f.shape[1] + start_time).reshape(-1, 1)
        feature_name = feature[i]
        x_poly = poly_feature.fit_transform(x)
        lin_reg = LinearRegression()
        lin_reg.fit(x_poly, track_shape_feature)
        coefficients = lin_reg.coef_
        fitted_function = lin_reg.predict(x_poly)
        autocorrelation_function = acf(track_shape_feature, nlags=nlags)
        for j in range(matrix_t_f.shape[0]):
            second_track_shape_feature = matrix_t_f[j]
            if j != i:
                cross_correlation_function = ccf(
                    track_shape_feature, second_track_shape_feature, nlags=nlags
                )

                all_cross_correlations.append(cross_correlation_function)
        avg_cross_correlation_function = np.mean(
            np.asarray(all_cross_correlations), axis=0
        )

        if feature_name in nth_generation_polynomial_coefficients:
            nth_generation_polynomials[feature_name].append(fitted_function)
            nth_generation_polynomial_coefficients[feature_name].append(coefficients)
            nth_generation_polynomial_time[feature_name].append(start_end_times)
            nth_generation_autocorrelation[feature_name].append(
                autocorrelation_function
            )
            nth_generation_crosscorrelation[feature_name].append(
                avg_cross_correlation_function
            )
            nth_generation_covariance[feature_name].append(cov_matrix[i])
            nth_generation_raw[feature_name].append(track_shape_feature)
        else:
            nth_generation_polynomial_coefficients[feature_name] = [coefficients]
            nth_generation_polynomial_time[feature_name] = [start_end_times]
            nth_generation_autocorrelation[feature_name] = [autocorrelation_function]
            nth_generation_crosscorrelation[feature_name] = [
                avg_cross_correlation_function
            ]
            nth_generation_polynomials[feature_name] = [fitted_function]
            nth_generation_covariance[feature_name] = [cov_matrix[i]]
            nth_generation_raw[feature_name] = [track_shape_feature]


def populate_daughter_tracklets(
    daughter_generations,
    tracks_dataframe,
    generation_id,
    nth_generation_life,
    nth_generation_polynomial_coefficients,
    nth_generation_polynomials,
    nth_generation_polynomial_time,
    nth_generation_autocorrelation,
    nth_generation_crosscorrelation,
    nth_generation_covariance,
    nth_generation_raw,
    shape_analysis=True,
    cross_analysis=False,
    nlags=50,
):
    for trackmate_track_id in daughter_generations[generation_id].keys():
        generation_daughters = daughter_generations[generation_id][trackmate_track_id]
        for daughters in generation_daughters:
            daughter_tracklet_id, start_time, end_time = daughters
            start_end_times = np.vstack((start_time, end_time))

            nth_generation_life.append(end_time - start_time)
            dividing_track_track_data = tracks_dataframe[
                (tracks_dataframe["Track ID"] == daughter_tracklet_id)
            ].sort_values(by="t")
            result = cell_fate_recipe(dividing_track_track_data)
            if result is not None:
                (
                    dividing_shape_dynamic_track_array,
                    dividing_shape_track_array,
                    dividing_dynamic_track_array,
                    dividing_covariance_shape_dynamic,
                    dividing_covariance_shape,
                    dividing_covariance_dynamic,
                    dividing_shape_dynamic_prediction_input,
                    dividing_shape_prediction_input,
                    dividing_dynamic_prediction_input,
                ) = result

                if shape_analysis:

                    track_array = dividing_shape_track_array
                    feature = SHAPE_FEATURES

                elif not shape_analysis and not cross_analysis:

                    track_array = dividing_dynamic_track_array
                    feature = DYNAMIC_FEATURES

                elif cross_analysis and not shape_analysis:

                    track_array = dividing_shape_dynamic_track_array
                    feature = SHAPE_DYNAMIC_FEATURES

                generic_polynomial_fits(
                    track_array,
                    start_time,
                    start_end_times,
                    nth_generation_polynomial_coefficients,
                    nth_generation_polynomials,
                    nth_generation_polynomial_time,
                    nth_generation_autocorrelation,
                    nth_generation_crosscorrelation,
                    nth_generation_covariance,
                    nth_generation_raw,
                    feature,
                    nlags=nlags,
                )


def create_cluster_plot(
    dataframe,
    cluster_type,
    cluster_distance_type,
    cluster_eucledian_distance_type,
    unique_col_names=[
        "Cluster_Label_Distances",
        "Cluster_Label_Eucledian_Distances",
        "Cluster_Label",
    ],
    negate_cluster_type=None,
    track_duration=0,
    show_plot=False,
    negate_cluster_distance_type=None,
    negate_cluster_eucledian_distance_type=None,
):

    if negate_cluster_type is None:
        cluster_columns = [col for col in dataframe.columns if cluster_type in col]
    else:
        cluster_columns = [
            col
            for col in dataframe.columns
            if cluster_type in col and negate_cluster_type not in col
        ]

    if negate_cluster_distance_type is None:
        cluster_distance_columns = [
            col for col in dataframe.columns if cluster_distance_type in col
        ]
    else:
        cluster_distance_columns = [
            col
            for col in dataframe.columns
            if cluster_distance_type in col and negate_cluster_distance_type not in col
        ]
    if negate_cluster_eucledian_distance_type is None:
        cluster_eucledian_distance_columns = [
            col for col in dataframe.columns if cluster_eucledian_distance_type in col
        ]
    else:
        cluster_eucledian_distance_columns = [
            col
            for col in dataframe.columns
            if cluster_eucledian_distance_type in col
            and negate_cluster_eucledian_distance_type not in col
            and "Eucledian" in col
        ]
    df_time_clusters_melted = dataframe[
        [
            "t",
            "z",
            "y",
            "x",
            "TrackMate Track ID",
            "Track ID",
            "Dividing",
            "Cell_Type",
            "Number_Dividing",
            "Track Duration",
            "MSD",
        ]
        + cluster_columns
        + cluster_distance_columns
        + cluster_eucledian_distance_columns
        + SHAPE_FEATURES
        + DYNAMIC_FEATURES
    ].copy()
    data = []
    tinit = 0
    for index, cluster_column in enumerate(cluster_columns):
        if index == len(cluster_columns) - 1:
            time_veto = dataframe["t"].max()
        else:
            time_veto = int(extract_number_from_string(cluster_columns[index + 1])[0])
        if index == 0:
            tstart = -1
            tend = time_veto
        else:
            tstart = tinit
            tend = time_veto
        filtered_df = df_time_clusters_melted[
            (df_time_clusters_melted["t"] > tstart)
            & (df_time_clusters_melted["t"] <= tend)
        ]
        data.extend(filtered_df[cluster_column])

        if re.search(r"\bShape\sDynamic\b", cluster_column):
            append_name = "Shape_Dynamic_"
        elif re.search(r"\bShape\b", cluster_column):
            append_name = "Shape_"
        elif re.search(r"\bDynamic\b", cluster_column):
            append_name = "Dynamic_"

        tinit = time_veto
    df_time_clusters_melted[append_name + unique_col_names[-1]] = data

    data = []
    eucledian_data = []
    tinit = 0
    for index, cluster_distance_column in enumerate(cluster_distance_columns):
        if index == len(cluster_distance_columns) - 1:
            time_veto = dataframe["t"].max()
        else:
            time_veto = int(
                extract_number_from_string(cluster_distance_columns[index + 1])[0]
            )
        if index == 0:
            tstart = -1
            tend = time_veto
        else:
            tstart = tinit
            tend = time_veto
        filtered_df = df_time_clusters_melted[
            (df_time_clusters_melted["t"] > tstart)
            & (df_time_clusters_melted["t"] <= tend)
        ]

        data.extend(filtered_df[cluster_distance_column].tolist())
        if re.search(r"\bShape\sDynamic\b", cluster_distance_column):
            append_name = "Shape_Dynamic_"
        elif re.search(r"\bShape\b", cluster_distance_column):
            append_name = "Shape_"
        elif re.search(r"\bDynamic\b", cluster_distance_column):
            append_name = "Dynamic_"

        tinit = time_veto

    for index, cluster_eucledian_distance_column in enumerate(
        cluster_eucledian_distance_columns
    ):
        if index == len(cluster_distance_columns) - 1:
            time_veto = dataframe["t"].max()
        else:
            time_veto = int(
                extract_number_from_string(cluster_distance_columns[index + 1])[0]
            )
        if index == 0:
            tstart = -1
            tend = time_veto
        else:
            tstart = tinit
            tend = time_veto
        filtered_df = df_time_clusters_melted[
            (df_time_clusters_melted["t"] > tstart)
            & (df_time_clusters_melted["t"] <= tend)
        ]

        eucledian_data.extend(filtered_df[cluster_eucledian_distance_column].tolist())
        tinit = time_veto

    df_time_clusters_melted[append_name + unique_col_names[0]] = data
    df_time_clusters_melted[append_name + unique_col_names[1]] = eucledian_data

    df_time_clusters_melted = df_time_clusters_melted.drop(columns=cluster_columns)
    df_time_clusters_melted = df_time_clusters_melted.drop(
        columns=cluster_distance_columns
    )
    df_time_clusters_melted = df_time_clusters_melted.drop(
        columns=cluster_eucledian_distance_columns
    )
    df_time_clusters_melted = df_time_clusters_melted.dropna(
        subset=append_name + unique_col_names[-1]
    )
    df_time_clusters_melted = df_time_clusters_melted.dropna(
        subset=append_name + unique_col_names[0]
    )
    df_time_clusters_melted = df_time_clusters_melted.dropna(
        subset=append_name + unique_col_names[1]
    )

    return df_time_clusters_melted


def extract_number_from_string(string):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, string)
    numbers = [float(match) for match in matches]
    return numbers


def extract_number_dividing(file_name):
    number_dividing = -1  # Default value if the pattern isn't found
    match = re.search(r"Number_Dividing_(\d+)\.npy", file_name)
    if match:
        number_dividing = int(match.group(1))
    return number_dividing


def extract_celltype(file_name):
    cell_type = -1
    match = re.search(r"Cell_Type_(\d+)\.npy", file_name)
    if match:
        cell_type = int(match.group(1))
    return cell_type


def cross_correlation_class(tracks_dataframe, cell_type_label=None):

    if cell_type_label is not None:
        tracks_dataframe = tracks_dataframe[
            (tracks_dataframe["Cell_Type_Label"] == cell_type_label)
        ]

    generation_max = tracks_dataframe["Generation ID"].max()
    sorted_dividing_dataframe = tracks_dataframe.sort_values(
        by="Track Duration", ascending=False
    )

    unique_trackmate_track_ids = sorted_dividing_dataframe[
        "TrackMate Track ID"
    ].unique()
    zero_gen_tracklets = {}
    daughter_generations = {i: {} for i in range(1, generation_max + 1)}
    get_zero_gen_daughter_generations(
        unique_trackmate_track_ids,
        tracks_dataframe,
        zero_gen_tracklets,
        daughter_generations,
    )

    zero_gen_dynamic_polynomials = {}
    zero_gen_dynamic_polynomial_coefficients = {}
    zero_gen_dynamic_raw = {}
    zero_gen_dynamic_autocorrelation = {}
    zero_gen_dynamic_crosscorrelation = {}
    zero_gen_dynamic_covariance = {}
    zero_gen_dynamic_polynomial_time = {}
    zero_gen_dynamic_life = []

    populate_zero_gen_tracklets(
        zero_gen_tracklets,
        tracks_dataframe,
        zero_gen_dynamic_life,
        zero_gen_dynamic_polynomial_coefficients,
        zero_gen_dynamic_polynomials,
        zero_gen_dynamic_polynomial_time,
        zero_gen_dynamic_autocorrelation,
        zero_gen_dynamic_crosscorrelation,
        zero_gen_dynamic_covariance,
        zero_gen_dynamic_raw,
        shape_analysis=False,
    )

    zero_gen_shape_polynomials = {}
    zero_gen_shape_polynomial_coefficients = {}
    zero_gen_shape_raw = {}
    zero_gen_shape_autocorrelation = {}
    zero_gen_shape_crosscorrelation = {}
    zero_gen_shape_covariance = {}
    zero_gen_shape_polynomial_time = {}
    zero_gen_shape_life = []

    populate_zero_gen_tracklets(
        zero_gen_tracklets,
        tracks_dataframe,
        zero_gen_shape_life,
        zero_gen_shape_polynomial_coefficients,
        zero_gen_shape_polynomials,
        zero_gen_shape_polynomial_time,
        zero_gen_shape_autocorrelation,
        zero_gen_shape_crosscorrelation,
        zero_gen_shape_covariance,
        zero_gen_shape_raw,
        shape_analysis=True,
    )

    N_shape_generation_polynomials = {}
    N_shape_generation_polynomial_coefficients = {}
    N_shape_generation_raw = {}
    N_shape_generation_autocorrelation = {}
    N_shape_generation_crosscorrelation = {}
    N_shape_generation_covariance = {}
    N_shape_generation_polynomial_time = {}
    N_shape_generation_life = []

    for generation_id in daughter_generations.keys():
        if generation_id >= 1:
            populate_daughter_tracklets(
                daughter_generations,
                tracks_dataframe,
                generation_id,
                N_shape_generation_life,
                N_shape_generation_polynomial_coefficients,
                N_shape_generation_polynomials,
                N_shape_generation_polynomial_time,
                N_shape_generation_autocorrelation,
                N_shape_generation_crosscorrelation,
                N_shape_generation_covariance,
                N_shape_generation_raw,
                shape_analysis=True,
            )

    N_dynamic_generation_polynomials = {}
    N_dynamic_generation_polynomial_coefficients = {}
    N_dynamic_generation_raw = {}
    N_dynamic_generation_autocorrelation = {}
    N_dynamic_generation_crosscorrelation = {}
    N_dynamic_generation_covariance = {}
    N_dynamic_generation_polynomial_time = {}
    N_dynamic_generation_life = []

    for generation_id in daughter_generations.keys():
        if generation_id >= 1:
            populate_daughter_tracklets(
                daughter_generations,
                tracks_dataframe,
                generation_id,
                N_dynamic_generation_life,
                N_dynamic_generation_polynomial_coefficients,
                N_dynamic_generation_polynomials,
                N_dynamic_generation_polynomial_time,
                N_dynamic_generation_autocorrelation,
                N_dynamic_generation_crosscorrelation,
                N_dynamic_generation_covariance,
                N_dynamic_generation_raw,
                shape_analysis=False,
            )

    zero_gen_dynamic_sigma_dict = {}
    zero_gen_dynamic_test_dict = {}
    zero_gen_dynamic_conccross = {}
    for (
        dynamic_feature,
        list_dynamic_crosscorrelation_functions,
    ) in zero_gen_dynamic_crosscorrelation.items():

        concatenated_crosscorrs = np.concatenate(
            [
                crosscorr[~np.isnan(crosscorr)]
                for crosscorr in list_dynamic_crosscorrelation_functions
            ]
        )
        mean, std_dev = norm.fit(concatenated_crosscorrs)
        zero_gen_dynamic_test_stat = anderson(concatenated_crosscorrs)
        zero_gen_dynamic_sigma_dict[dynamic_feature] = std_dev
        zero_gen_dynamic_test_dict[dynamic_feature] = zero_gen_dynamic_test_stat
        zero_gen_dynamic_conccross[dynamic_feature] = concatenated_crosscorrs

    zero_gen_shape_sigma_dict = {}
    zero_gen_shape_test_dict = {}
    zero_gen_shape_conccross = {}
    for (
        shape_feature,
        list_shape_crosscorrelation_functions,
    ) in zero_gen_shape_crosscorrelation.items():

        concatenated_crosscorrs = np.concatenate(
            [
                crosscorr[~np.isnan(crosscorr)]
                for crosscorr in list_shape_crosscorrelation_functions
            ]
        )
        mean, std_dev = norm.fit(concatenated_crosscorrs)
        zero_gen_shape_test_stat = anderson(concatenated_crosscorrs)

        zero_gen_shape_sigma_dict[shape_feature] = std_dev
        zero_gen_shape_test_dict[shape_feature] = zero_gen_shape_test_stat
        zero_gen_shape_conccross[shape_feature] = concatenated_crosscorrs

    N_gen_dynamic_sigma_dict = {}
    N_gen_dynamic_test_dict = {}
    N_gen_dynamic_conccross = {}
    for (
        dynamic_feature,
        list_dynamic_crosscorrelation_functions,
    ) in N_dynamic_generation_crosscorrelation.items():

        concatenated_crosscorrs = np.concatenate(
            [
                crosscorr[~np.isnan(crosscorr)]
                for crosscorr in list_dynamic_crosscorrelation_functions
            ]
        )
        mean, std_dev = norm.fit(concatenated_crosscorrs)
        N_gen_dynamic_test_stat = anderson(concatenated_crosscorrs)
        N_gen_dynamic_sigma_dict[dynamic_feature] = std_dev
        N_gen_dynamic_test_dict[dynamic_feature] = N_gen_dynamic_test_stat
        N_gen_dynamic_conccross[dynamic_feature] = concatenated_crosscorrs

    N_gen_shape_sigma_dict = {}
    N_gen_shape_test_dict = {}
    N_gen_shape_conccross = {}
    for (
        shape_feature,
        list_shape_crosscorrelation_functions,
    ) in N_shape_generation_crosscorrelation.items():

        concatenated_crosscorrs = np.concatenate(
            [
                crosscorr[~np.isnan(crosscorr)]
                for crosscorr in list_shape_crosscorrelation_functions
            ]
        )
        mean, std_dev = norm.fit(concatenated_crosscorrs)
        N_gen_shape_test_stat = anderson(concatenated_crosscorrs)
        N_gen_shape_sigma_dict[shape_feature] = std_dev
        N_gen_shape_test_dict[shape_feature] = N_gen_shape_test_stat
        N_gen_shape_conccross[shape_feature] = concatenated_crosscorrs

    return (
        zero_gen_dynamic_conccross,
        zero_gen_shape_conccross,
        N_gen_dynamic_conccross,
        N_gen_shape_conccross,
        zero_gen_dynamic_sigma_dict,
        zero_gen_shape_sigma_dict,
        N_gen_dynamic_sigma_dict,
        N_gen_shape_sigma_dict,
        zero_gen_dynamic_test_dict,
        zero_gen_shape_test_dict,
        N_gen_dynamic_test_dict,
        N_gen_shape_test_dict,
    )


def plot_at_mitosis_time(matrix_directory, save_dir, dataset_name, channel):

    files = os.listdir(matrix_directory)

    sorted_files = natsorted(
        [
            file
            for file in files
            if file.endswith(".npy") and "data_at_mitosis_time" in file
        ],
        key=lambda x: (
            "_Dynamic Cluster" in x,
            "_Shape Dynamic Cluster" in x,
            "_Shape Cluster" in x,
            x,
        ),
    )

    excluded_keys = [
        "Track ID",
        "t",
        "z",
        "y",
        "x",
        "Unnamed: 0",
        "Unnamed",
        "Track Duration",
        "Generation ID",
        "TrackMate Track ID",
        "Tracklet Number ID",
        "Tracklet_ID",
        "Unique_ID",
        "Track_ID",
    ]

    for file_name in sorted_files:

        all_split_data = np.load(
            os.path.join(matrix_directory, file_name), allow_pickle=True
        )

        grouped_data = {}
        for data_dict in all_split_data:
            number_times_divided = data_dict.get("Number Times Divided")
            for key, value in data_dict.items():
                if key not in excluded_keys:
                    if key not in grouped_data:
                        grouped_data[key] = {number_times_divided: [value]}
                    else:
                        if number_times_divided not in grouped_data[key]:
                            grouped_data[key][number_times_divided] = [value]
                        else:
                            grouped_data[key][number_times_divided].append(value)

        for property_name, property_data in grouped_data.items():
            plt.figure(figsize=(10, 6))
            for number_times_divided, values in property_data.items():
                sns.histplot(
                    values,
                    label=f"Number Times Divided {number_times_divided}",
                    kde=True,
                    bins=20,
                    edgecolor="black",
                    alpha=0.5,
                )
            plt.xlabel("Property Values")
            plt.ylabel("Frequency")
            plt.title(f"{property_name}")

            fig_name = (
                f"{dataset_name}_{channel}_{property_name}_at_mitosis_distribution.png"
            )
            plt.savefig(os.path.join(save_dir, fig_name), dpi=300, bbox_inches="tight")
            plt.show()


def plot_histograms_for_groups(
    matrix_directory, save_dir, dataset_name, channel, name="all", plot_show=True
):

    files = os.listdir(matrix_directory)
    sorted_files = natsorted(
        [
            file
            for file in files
            if file.endswith(".npy")
            and "Number_Dividing" in file
            and "Number_DividingNumber_Dividing"
            and "DividingNumber_Dividing" not in file
        ]
    )
    groups = set()

    for file_name in sorted_files:
        group_name = file_name.split("Number_Dividing")[0]

        groups.add(group_name)

    for group_name in groups:
        if len(group_name) > 0:
            plt.figure(figsize=(8, 6))
            group_files = [file for file in sorted_files if group_name in file]

            for file_name in group_files:
                file_path = os.path.join(matrix_directory, file_name)
                number_dividing = extract_number_dividing(file_name)

                data = np.load(file_path, allow_pickle=True)
                sns.histplot(
                    data,
                    alpha=0.5,
                    kde=True,
                    label=f"Number_Dividing: {number_dividing}",
                )

            plt.xlabel("Value")
            plt.ylabel("Counts")
            simplified_group_name = re.search(r"_(.*?)_Number_Dividing", group_name)
            simplified_group_name = (
                simplified_group_name.group(1) if simplified_group_name else group_name
            )
            plt.title(f"{simplified_group_name}")
            plt.legend()
            fig_name = f"{channel}{group_name}_{name}_distribution.png"
            plt.savefig(os.path.join(save_dir, fig_name), dpi=300, bbox_inches="tight")
            if plot_show:
                plt.show()


def plot_histograms_for_cell_type_groups(
    matrix_directory,
    save_dir,
    dataset_name,
    channel,
    label_dict=None,
    name="all",
    plot_show=True,
):

    files = os.listdir(matrix_directory)
    sorted_files = natsorted(
        [
            file
            for file in files
            if file.endswith(".npy")
            and "Cell_Type" in file
            and "Cell_Type_LabelCell_Type"
            and "Cell_TypeCell_Type" not in file
        ]
    )
    groups = set()

    for file_name in sorted_files:
        group_name = file_name.split("Cell_Type")[0]

        groups.add(group_name)

    for group_name in groups:
        if len(group_name) > 0:
            plt.figure(figsize=(8, 6))
            group_files = [file for file in sorted_files if group_name in file]

            for file_name in group_files:
                file_path = os.path.join(matrix_directory, file_name)
                cell_type = extract_celltype(file_name)
                if label_dict is not None:
                    cell_type_name = label_dict[cell_type]
                else:
                    cell_type_name = cell_type
                data = np.load(file_path, allow_pickle=True)
                sns.histplot(
                    data, alpha=0.5, kde=True, label=f"Cell_Type: {cell_type_name}"
                )

            plt.xlabel("Value")
            plt.ylabel("Counts")
            simplified_group_name = re.search(r"_(.*?)_Cell_Type", group_name)
            simplified_group_name = (
                simplified_group_name.group(1) if simplified_group_name else group_name
            )
            plt.title(f"{simplified_group_name}")
            plt.legend()
            fig_name = f"{channel}{group_name}_{name}_distribution.png"
            plt.savefig(os.path.join(save_dir, fig_name), dpi=300, bbox_inches="tight")
            if plot_show:
                plt.show()


def create_movie(df, column, time_plot):

    fig = plt.figure(figsize=(10, 12))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212, projection="polar")
    filtered_dataframe = df[df["t"] == time_plot]
    x, y, z = filtered_dataframe["x"], filtered_dataframe["y"], filtered_dataframe["z"]

    angles_deg = filtered_dataframe[column].values
    angle_rad = np.deg2rad(angles_deg)

    arrow_length = 20
    norm = plt.Normalize(angles_deg.min(), angles_deg.max())
    cmap = cm.viridis

    if "Y" in column:
        x_end, y_end = arrow_length * np.sin(angle_rad), arrow_length * np.cos(
            angle_rad
        )

        ax0.quiver(x, y, x_end, y_end, linewidth=2, color=cmap(norm(angles_deg)))

        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        ax0.set_title(f"Orientation Plot {column}")
        plt.colorbar(ax0.scatter(x, y, c=angles_deg))
    if "X" in column:
        x_end, y_end = arrow_length * np.cos(angle_rad), arrow_length * np.sin(
            angle_rad
        )

        ax0.quiver(x, y, x_end, y_end, linewidth=2, color=cmap(norm(angles_deg)))

        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        ax0.set_title(f"Orientation Plot {column}")
        plt.colorbar(ax0.scatter(x, y, c=angles_deg))
    if "Z" in column:
        x_end, z_end = arrow_length * np.sin(angle_rad), arrow_length * np.cos(
            angle_rad
        )

        ax0.quiver(x, z, x_end, z_end, linewidth=2, color=cmap(norm(angles_deg)))

        ax0.set_xlabel("X")
        ax0.set_ylabel("Z")
        ax0.set_title(f"Orientation Plot {column}")
        plt.colorbar(ax0.scatter(x, z, c=angles_deg))
    hist, bins = np.histogram(angle_rad)

    ax1.bar(bins[:-1], hist, width=2 * np.pi / len(bins), color="skyblue")
    ax1.set_theta_direction(1)
    ax1.set_theta_zero_location("E")
    ax1.set_thetamin(0)
    ax1.set_thetamax(180)

    ax1.set_title(f"Circular Plot of {column}")

    fig.canvas.draw()
    plot_array = np.array(fig.canvas.renderer.buffer_rgba())

    plt.close()

    return plot_array


def create_video(frames, video_filename, fps=1):
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for frame_params in frames:
            frame = create_movie(*frame_params)
            writer.append_data(frame)


def angular_plot(global_shape_dynamic_dataframe, column="Radial_Angle_Z", time_plot=0):

    filtered_dataframe = global_shape_dynamic_dataframe[
        global_shape_dynamic_dataframe["t"] == time_plot
    ]
    x, y, z = filtered_dataframe["x"], filtered_dataframe["y"], filtered_dataframe["z"]

    angles_deg = filtered_dataframe[column].values
    angle_rad = np.deg2rad(angles_deg)

    arrow_length = 20
    norm = plt.Normalize(angles_deg.min(), angles_deg.max())
    cmap = cm.viridis
    plt.figure(figsize=(16, 16))

    if "Y" in column:
        x_end, y_end = arrow_length * np.sin(angle_rad), arrow_length * np.cos(
            angle_rad
        )

        plt.quiver(x, y, x_end, y_end, linewidth=2, color=cmap(norm(angles_deg)))

        plt.xlabel("Y")
        plt.ylabel("X")
        plt.title(f"Orientation Plot {column}")
        plt.colorbar(plt.scatter(x, y, c=angles_deg))
        plt.show()
    if "X" in column:
        x_end, y_end = arrow_length * np.cos(angle_rad), arrow_length * np.sin(
            angle_rad
        )

        plt.quiver(x, y, x_end, y_end, linewidth=2, color=cmap(norm(angles_deg)))

        plt.xlabel("Y")
        plt.ylabel("X")
        plt.title(f"Orientation Plot {column}")
        plt.colorbar(plt.scatter(x, y, c=angles_deg))
        plt.show()
    if "Z" in column:
        x_end, z_end = arrow_length * np.sin(angle_rad), arrow_length * np.cos(
            angle_rad
        )

        plt.quiver(x, z, x_end, z_end, linewidth=2, color=cmap(norm(angles_deg)))

        plt.xlabel("Z")
        plt.ylabel("X")
        plt.title(f"Orientation Plot {column}")
        plt.colorbar(plt.scatter(x, z, c=angles_deg))
        plt.show()
    hist, bins = np.histogram(angle_rad)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(bins[:-1], hist, width=2 * np.pi / len(bins), color="skyblue")
    ax.set_theta_direction(1)
    ax.set_theta_zero_location("E")
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_title(f"Circular Plot of {column}")

    plt.show()
    clear_output(wait=True)


def normalize_list(lst):
    mean_val = np.mean(lst)
    std_val = np.std(lst)
    return [(x - mean_val) / std_val for x in lst]


def microdomain_plot(
    time,
    df,
    time_delta=10,
    cluster_type="Shape_Cluster_Label_Distances",
    cluster_eucledian_type="Shape_Cluster_Label_Eucledian_Distances",
    cluster_label_type="Shape_Cluster_Label",
    feature_x="x",
    feature_y="y",
    show=True,
):

    x_list = []
    y_list = []
    cluster_label_list = []
    cluster_label_dist_list = []
    cluster_label_eucledian_dist_list = []
    dividing_list = []
    for t in range(time, min(time + time_delta, int(df["t"].max()))):
        filtered_df = df[(df["t"] == t)]

        filtered_df = filtered_df[filtered_df[cluster_type].notna()]

        unique_labels = filtered_df[cluster_label_type].dropna().unique()
        for label in unique_labels:
            track_ids_with_label = filtered_df[
                filtered_df[cluster_label_type] == label
            ]["Track ID"].unique()
            for plot_idx, track_id in enumerate(track_ids_with_label):
                track_data = filtered_df[filtered_df["Track ID"] == track_id]

                cluster_label_dist = track_data[cluster_type].tolist()[0]

                cluster_label_eucledian_dist = track_data[
                    cluster_eucledian_type
                ].tolist()[0]

                x_list.append(track_data[feature_x].tolist()[0])
                y_list.append(track_data[feature_y].tolist()[0])
                cluster_label_list.append(track_data[cluster_label_type].tolist()[0])
                cluster_label_dist_list.append(cluster_label_dist)
                cluster_label_eucledian_dist_list.append(cluster_label_eucledian_dist)
                dividing_list.append(track_data["Dividing"].tolist()[0])

    color_dict = {
        "x": x_list,
        "y": y_list,
        "cluster_label": cluster_label_list,
        "cluster_label_dist": cluster_label_dist_list,
        "cluster_label_eucledian_dist": cluster_label_eucledian_dist_list,
        "dividing": dividing_list,
    }

    x_scatter = color_dict["x"]
    y_scatter = color_dict["y"]
    label_c_scatter = color_dict["cluster_label"]
    c_scatter = color_dict["cluster_label_dist"]
    eucledian_c_scatter = color_dict["cluster_label_eucledian_dist"]
    pure_dividing_c_scatter = color_dict["dividing"]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_scatter, y_scatter, c=label_c_scatter, cmap="viridis")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(cluster_label_type)
    plt.colorbar(scatter, label=cluster_label_type)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_scatter, y_scatter, c=c_scatter, cmap="viridis")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(cluster_type)
    plt.colorbar(scatter, label=cluster_type)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_scatter, y_scatter, c=eucledian_c_scatter, cmap="viridis")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(cluster_eucledian_type)
    plt.colorbar(scatter, label=cluster_eucledian_type)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x_scatter, y_scatter, c=pure_dividing_c_scatter, cmap="viridis"
    )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title("Dividing")
    plt.colorbar(scatter, label="Dividing")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return color_dict


def create_microdomain_movie(
    time,
    df,
    feature_x,
    feature_y,
    cluster_type,
    cluster_eucledian_type,
    cluster_label_type,
    time_delta=10,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    color_dict = microdomain_plot(
        time=time,
        df=df,
        feature_x=feature_x,
        feature_y=feature_y,
        time_delta=time_delta,
        cluster_type=cluster_type,
        cluster_eucledian_type=cluster_eucledian_type,
        cluster_label_type=cluster_label_type,
        show=False,
    )

    # Plotting the first scatter plot
    scatter1 = axes[0, 0].scatter(
        color_dict["x"], color_dict["y"], c=color_dict["cluster_label"], cmap="viridis"
    )
    axes[0, 0].set_xlabel(feature_x)
    axes[0, 0].set_ylabel(feature_y)
    axes[0, 0].set_title("cluster_label")
    plt.colorbar(scatter1, ax=axes[0, 0], label="cluster_label")

    # Plotting the second scatter plot
    scatter2 = axes[0, 1].scatter(
        color_dict["x"],
        color_dict["y"],
        c=color_dict["cluster_label_dist"],
        cmap="viridis",
    )
    axes[0, 1].set_xlabel(feature_x)
    axes[0, 1].set_ylabel(feature_y)
    axes[0, 1].set_title(cluster_type)
    plt.colorbar(scatter2, ax=axes[0, 1], label=cluster_type)

    scatter3 = axes[1, 0].scatter(
        color_dict["x"],
        color_dict["y"],
        c=color_dict["cluster_label_eucledian_dist"],
        cmap="viridis",
    )
    axes[1, 0].set_xlabel(feature_x)
    axes[1, 0].set_ylabel(feature_y)
    axes[1, 0].set_title(cluster_eucledian_type)
    plt.colorbar(scatter3, ax=axes[1, 0], label=cluster_eucledian_type)

    num_dividing_c_scatter = color_dict["dividing"]

    scatter4 = axes[1, 1].scatter(
        color_dict["x"], color_dict["y"], c=num_dividing_c_scatter, cmap="viridis"
    )
    axes[1, 1].set_xlabel(feature_x)
    axes[1, 1].set_ylabel(feature_y)
    axes[1, 1].set_title("dividing")
    plt.colorbar(scatter4, ax=axes[1, 1], label=cluster_eucledian_type)

    plt.tight_layout()

    fig.canvas.draw()
    plot_array = np.array(fig.canvas.renderer.buffer_rgba())

    plt.close()

    return plot_array


def create_microdomain_video(frames, video_filename, fps=1):
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for frame_params in frames:
            frame = create_microdomain_movie(*frame_params)
            writer.append_data(frame)


def find_closest_key(test_point, unique_dict, time_veto, space_veto):
    closest_key = None
    spot_id = None
    min_distance = float("inf")

    t_test, z_test, y_test, x_test = test_point
    for key in unique_dict.keys():
        t_key, z_key, y_key, x_key = key

        time_distance = abs(t_key - t_test)
        space_distance = np.sqrt(
            (z_key - z_test) ** 2 + (y_key - y_test) ** 2 + (x_key - x_test) ** 2
        )

        if time_distance <= time_veto and space_distance <= space_veto:
            if time_distance + space_distance < min_distance:
                min_distance = time_distance + space_distance
                closest_key = key
    if closest_key is not None:
        spot_id = unique_dict[closest_key]
    return spot_id


def update_distance_cluster_plot(time, df, feature_x="x", feature_y="y"):
    filtered_df = df[(df["t"] == time)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cluster_types = [
        f"Shape Dynamic Intra Cluster Distance_{time}",
        f"Shape Intra Cluster Distance_{time}",
        f"Dynamic Intra Cluster Distance_{time}",
    ]
    cluster_labels = [
        f"Shape Dynamic Cluster_{time}",
        f"Shape Cluster_{time}",
        f"Dynamic Cluster_{time}",
    ]
    for idx, cluster_type in enumerate(cluster_types):
        filtered_df = filtered_df[filtered_df[cluster_type].notna()]
        color_dict = {}
        cluster_label_type = cluster_labels[idx]
        unique_labels = filtered_df[cluster_label_type].unique()
        for label in unique_labels:
            track_ids_with_label = filtered_df[
                filtered_df[cluster_label_type] == label
            ]["Track ID"].unique()
            for plot_idx, track_id in enumerate(track_ids_with_label):
                track_data = filtered_df[filtered_df["Track ID"] == track_id]

                t_list = track_data["t"].tolist()

                cluster_label_dist = track_data[cluster_type].tolist()[0]

                cluster_label_dist = np.full_like(
                    np.asarray(track_ids_with_label), cluster_label_dist
                )
                x_y_dict = {
                    "t": t_list,
                    "x": track_data[feature_x].tolist(),
                    "y": track_data[feature_y].tolist(),
                    "cluster_label": track_data[cluster_label_type].tolist(),
                    "cluster_label_dist": cluster_label_dist,
                }
                color_index = x_y_dict["cluster_label_dist"][plot_idx]
                color = [color_index for _ in range(len(x_y_dict["x"]))]
                if label in color_dict:
                    color_dict[label][0].extend(x_y_dict["x"])
                    color_dict[label][1].extend(x_y_dict["y"])
                    color_dict[label][2].extend(color)
                else:
                    color_dict[label] = [x_y_dict["x"], x_y_dict["y"], color]

        x_scatter = [
            item
            for sublist in [color_dict[label][0] for label in unique_labels]
            for item in sublist
        ]
        y_scatter = [
            item
            for sublist in [color_dict[label][1] for label in unique_labels]
            for item in sublist
        ]
        c_scatter = [
            item
            for sublist in [color_dict[label][2] for label in unique_labels]
            for item in sublist
        ]
        scatter = axs[idx].scatter(x_scatter, y_scatter, c=c_scatter, cmap="viridis")

        plt.colorbar(scatter, ax=axs[idx], label="Cluster Label Distance")

        axs[idx].set_xlabel(feature_x)
        axs[idx].set_ylabel(feature_y)
        axs[idx].set_title(cluster_type)

    plt.tight_layout()
    plt.show()


def update_eucledian_distance_cluster_plot(time, df, feature_x="x", feature_y="y"):
    filtered_df = df[(df["t"] == time)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cluster_types = [
        f"Shape Dynamic Intra Cluster Eucledian Distance_{time}",
        f"Shape Intra Cluster Eucledian Distance_{time}",
        f"Dynamic Intra Cluster Eucledian Distance_{time}",
    ]
    cluster_labels = [
        f"Shape Dynamic Cluster_{time}",
        f"Shape Cluster_{time}",
        f"Dynamic Cluster_{time}",
    ]
    for idx, cluster_type in enumerate(cluster_types):
        filtered_df = filtered_df[filtered_df[cluster_type].notna()]
        color_dict = {}
        cluster_label_type = cluster_labels[idx]
        unique_labels = filtered_df[cluster_label_type].unique()
        for label in unique_labels:
            track_ids_with_label = filtered_df[
                filtered_df[cluster_label_type] == label
            ]["Track ID"].unique()
            for plot_idx, track_id in enumerate(track_ids_with_label):
                track_data = filtered_df[filtered_df["Track ID"] == track_id]

                t_list = track_data["t"].tolist()

                cluster_label_dist = track_data[cluster_type].tolist()[0]

                cluster_label_dist = np.full_like(
                    np.asarray(track_ids_with_label), cluster_label_dist
                )
                x_y_dict = {
                    "t": t_list,
                    "x": track_data[feature_x].tolist(),
                    "y": track_data[feature_y].tolist(),
                    "cluster_label": track_data[cluster_label_type].tolist(),
                    "cluster_label_eucledian_dist": cluster_label_dist,
                }
                color_index = x_y_dict["cluster_label_eucledian_dist"][plot_idx]

                color = [color_index for _ in range(len(x_y_dict["x"]))]
                if label in color_dict:
                    color_dict[label][0].extend(x_y_dict["x"])
                    color_dict[label][1].extend(x_y_dict["y"])
                    color_dict[label][2].extend(color)
                else:
                    color_dict[label] = [x_y_dict["x"], x_y_dict["y"], color]

        x_scatter = [
            item
            for sublist in [color_dict[label][0] for label in unique_labels]
            for item in sublist
        ]
        y_scatter = [
            item
            for sublist in [color_dict[label][1] for label in unique_labels]
            for item in sublist
        ]
        c_scatter = [
            item
            for sublist in [color_dict[label][2] for label in unique_labels]
            for item in sublist
        ]

        scatter = axs[idx].scatter(x_scatter, y_scatter, c=c_scatter, cmap="viridis")

        plt.colorbar(scatter, ax=axs[idx], label="Cluster Label Eucledian Distance")

        axs[idx].set_xlabel(feature_x)
        axs[idx].set_ylabel(feature_y)
        axs[idx].set_title(cluster_type)

    plt.tight_layout()
    plt.show()


def update_cluster_plot(time, df, time_delta=0):

    filtered_df = df[(df["t"] >= time) & (df["t"] <= time + time_delta)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    cluster_types = [
        f"Shape Dynamic Cluster_{time}",
        f"Shape Cluster_{time}",
        f"Dynamic Cluster_{time}",
    ]

    for idx, cluster_type in enumerate(cluster_types):

        scatter = axs[idx].scatter(
            filtered_df["x"],
            filtered_df["y"],
            c=filtered_df[f"{cluster_type}"],
            cmap="viridis",
            alpha=0.7,
        )
        axs[idx].set_title(f"{cluster_type} Cluster Label")
        axs[idx].set_xlabel("X")
        axs[idx].set_ylabel("Y")
        fig.colorbar(scatter, ax=axs[idx], label="Cluster Label")

    plt.tight_layout()
    plt.show()


def sample_subarrays(data, tracklet_length, total_duration):

    max_start_index = total_duration - tracklet_length
    start_indices = random.sample(range(max_start_index), max_start_index)

    subarrays = []
    for start_index in start_indices:
        end_index = start_index + tracklet_length
        if end_index <= total_duration:
            sub_data = data[start_index:end_index, :]
            if sub_data.shape[0] == tracklet_length:
                subarrays.append(sub_data)

    return subarrays


def make_prediction(input_data, model, device):

    model = model.to(device)
    with torch.no_grad():
        input_tensor = (
            torch.tensor(input_data).unsqueeze(0).permute(0, 2, 1).float()
        ).to(device)
        model_predictions = model(input_tensor)
        probabilities = torch.softmax(model_predictions[0], dim=0)
        _, predicted_class = torch.max(probabilities, 0)
    return predicted_class.item()


def get_most_frequent_prediction(predictions):

    prediction_counts = Counter(predictions)
    try:
        most_common_prediction, count = prediction_counts.most_common(1)[0]

        return most_common_prediction
    except IndexError:
        return None


def weighted_prediction(predictions, weights):
    """
    Calculate the most frequent prediction with weighting by tracklet length.
    """
    weighted_counts = Counter()
    for prediction, weight in zip(predictions, weights):
        weighted_counts[prediction] += weight

    most_common_prediction, _ = weighted_counts.most_common(1)[0]
    return most_common_prediction


def vision_inception_model_prediction(
    dataframe,
    trackmate_id,
    raw_image,
    class_map,
    model,
    device="cpu",
    crop_size=(25, 8, 128, 128),
):
    """
    Generate predictions for an inception-style vision model based on patches around each point in a tracklet.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing track information.
        trackmate_id (int): The TrackMate track ID for which to generate predictions.
        tracklet_length (int): The number of time points in each tracklet.
        raw_image (np.array): The raw image from which patches are extracted.
        class_map (dict): Mapping of class indices to labels.
        model (torch.nn.Module): The model to use for predictions.
        device (str): Device for running predictions, 'cpu' or 'cuda'.
        crop_size (tuple): Size of the crop around each point (imagesizex, imagesizey, imagesizez).

    Returns:
        predictions (list): Predicted class labels for each tracklet.
        weights (list): Prediction confidence or logits for each tracklet.
    """
    model = model.to(device)
    model.eval()
    sub_trackmate_dataframe = dataframe[dataframe["TrackMate Track ID"] == trackmate_id]

    # Extract the dimensions of the crop
    imagesizet, sizez, sizex, sizey = crop_size
    tracklet_predictions = []
    tracklet_weights = []

    for tracklet_id in sub_trackmate_dataframe["Track ID"].unique():
        tracklet_sub_dataframe = sub_trackmate_dataframe[
            sub_trackmate_dataframe["Track ID"] == tracklet_id
        ]

        sub_trackmate_dataframe = tracklet_sub_dataframe.sort_values(by="t")
        total_duration = sub_trackmate_dataframe["Track Duration"].max()
        tracklet_blocks = []
        for i in range(0, len(sub_trackmate_dataframe), imagesizet):
            tracklet_block = sub_trackmate_dataframe.iloc[i : i + imagesizet][
                ["t", "z", "y", "x"]
            ].values
            if len(tracklet_block) == imagesizet:
                tracklet_blocks.append(tracklet_block)

        for tracklet_block in tracklet_blocks:
            stitched_volume = []
            for (t, z, y, x) in tracklet_block:
                small_image = raw_image[int(t)]

                if (
                    x > sizex / 2
                    and z > sizez / 2
                    and y > sizey / 2
                    and z + int(sizez / 2) < raw_image.shape[1]
                    and y + int(sizey / 2) < raw_image.shape[2]
                    and x + int(sizex / 2) < raw_image.shape[3]
                    and t < raw_image.shape[0]
                ):
                    crop_xminus = x - int(sizex / 2)
                    crop_xplus = x + int(sizex / 2)
                    crop_yminus = y - int(sizey / 2)
                    crop_yplus = y + int(sizey / 2)
                    crop_zminus = z - int(sizez / 2)
                    crop_zplus = z + int(sizez / 2)
                    region = (
                        slice(int(crop_zminus), int(crop_zplus)),
                        slice(int(crop_yminus), int(crop_yplus)),
                        slice(int(crop_xminus), int(crop_xplus)),
                    )

                    crop_image = small_image[region]
                    stitched_volume.append(crop_image)
            stitched_volume = np.stack(stitched_volume, axis=0)
            if stitched_volume.shape[0] == imagesizet:
                with torch.no_grad():
                    prediction_vector = model(
                        torch.unsqueeze(
                            torch.tensor(stitched_volume, dtype=torch.float32), dim=0
                        )
                    )

                class_logits = prediction_vector[0, : len(class_map)]

                most_frequent_prediction = get_most_frequent_prediction(class_logits)
                if most_frequent_prediction is not None:
                    most_predicted_class = class_map[int(most_frequent_prediction)]
                    tracklet_predictions.append(most_predicted_class)
                    tracklet_weights.append(total_duration)

    if tracklet_predictions:
        final_weighted_prediction = weighted_prediction(
            tracklet_predictions, tracklet_weights
        )
        return final_weighted_prediction

    else:
        return "UnClassified"


def inception_dual_model_prediction(
    dataframe,
    second_dataframe,
    trackmate_id,
    tracklet_length,
    class_map,
    dual_morphodynamic_model=None,
    single_morphodynamic_model=None,
    device="cpu",
):
    sub_trackmate_dataframe = dataframe[dataframe["TrackMate Track ID"] == trackmate_id]
    sub_secondtrackmate_dataframe = second_dataframe[
        second_dataframe["TrackMate Track ID"] == trackmate_id
    ]

    tracklet_predictions = []
    tracklet_weights = []

    for tracklet_id in sub_trackmate_dataframe["Track ID"].unique():
        tracklet_sub_dataframe = sub_trackmate_dataframe[
            sub_trackmate_dataframe["Track ID"] == tracklet_id
        ]
        second_tracklet_sub_dataframe = sub_secondtrackmate_dataframe[
            sub_secondtrackmate_dataframe["Track ID"] == tracklet_id
        ]

        # sub_dataframe_dynamic = tracklet_sub_dataframe[DYNAMIC_FEATURES].values
        sub_dataframe_shape = tracklet_sub_dataframe[SHAPE_FEATURES].values
        sub_dataframe_morpho = tracklet_sub_dataframe[SHAPE_DYNAMIC_FEATURES].values

        sub_second_dataframe_shape = second_tracklet_sub_dataframe[
            SHAPE_FEATURES
        ].values
        total_duration = tracklet_sub_dataframe["Track Duration"].max()
        if (
            sub_dataframe_shape.shape[0] == sub_second_dataframe_shape.shape[0]
            and dual_morphodynamic_model is not None
        ):

            combined_morpho = np.concatenate(
                (sub_dataframe_morpho, sub_second_dataframe_shape), axis=-1
            )
            sub_combined_arrays_morpho = sample_subarrays(
                combined_morpho, tracklet_length, total_duration
            )
            dual_morpho_predictions = []
            for sub_array in sub_combined_arrays_morpho:
                predicted_class = make_prediction(
                    sub_array, dual_morphodynamic_model, device
                )
                dual_morpho_predictions.append(predicted_class)

            most_frequent_prediction = get_most_frequent_prediction(
                dual_morpho_predictions
            )

        else:

            sub_arrays_morpho = sample_subarrays(
                sub_dataframe_morpho, tracklet_length, total_duration
            )
            morpho_predictions = []
            for sub_array in sub_arrays_morpho:
                predicted_class = make_prediction(
                    sub_array, single_morphodynamic_model, device
                )
                morpho_predictions.append(predicted_class)

            most_frequent_prediction = get_most_frequent_prediction(morpho_predictions)

        if most_frequent_prediction is not None:
            most_predicted_class = class_map[int(most_frequent_prediction)]
            tracklet_predictions.append(most_predicted_class)
            tracklet_weights.append(total_duration)

    if tracklet_predictions:
        final_weighted_prediction = weighted_prediction(
            tracklet_predictions, tracklet_weights
        )
        return final_weighted_prediction
    else:
        return "UnClassified"


def inception_model_prediction(
    dataframe,
    trackmate_id,
    tracklet_length,
    class_map,
    dynamic_model=None,
    shape_model=None,
    morphodynamic_model=None,
    device="cpu",
):

    sub_trackmate_dataframe = dataframe[dataframe["TrackMate Track ID"] == trackmate_id]
    tracklet_predictions = []
    tracklet_weights = []

    for tracklet_id in sub_trackmate_dataframe["Track ID"].unique():
        tracklet_sub_dataframe = sub_trackmate_dataframe[
            sub_trackmate_dataframe["Track ID"] == tracklet_id
        ]

        sub_dataframe_dynamic = tracklet_sub_dataframe[DYNAMIC_FEATURES].values
        sub_dataframe_shape = tracklet_sub_dataframe[SHAPE_FEATURES].values
        sub_dataframe_morpho = tracklet_sub_dataframe[SHAPE_DYNAMIC_FEATURES].values

        total_duration = tracklet_sub_dataframe["Track Duration"].max()

        sub_arrays_shape = sample_subarrays(
            sub_dataframe_shape, tracklet_length, total_duration
        )
        sub_arrays_dynamic = sample_subarrays(
            sub_dataframe_dynamic, tracklet_length, total_duration
        )

        sub_arrays_morpho = sample_subarrays(
            sub_dataframe_morpho, tracklet_length, total_duration
        )

        shape_predictions = []
        if shape_model is not None:
            for sub_array in sub_arrays_shape:
                predicted_class = make_prediction(sub_array, shape_model, device)
                shape_predictions.append(predicted_class)

        dynamic_predictions = []
        if dynamic_model is not None:
            for sub_array in sub_arrays_dynamic:
                predicted_class = make_prediction(sub_array, dynamic_model, device)
                dynamic_predictions.append(predicted_class)

        morpho_predictions = []
        if morphodynamic_model is not None:
            for sub_array in sub_arrays_morpho:
                predicted_class = make_prediction(
                    sub_array, morphodynamic_model, device
                )
                morpho_predictions.append(predicted_class)

        if morphodynamic_model is None:
            final_predictions = shape_predictions + dynamic_predictions
        else:
            final_predictions = morpho_predictions
        most_frequent_prediction = get_most_frequent_prediction(final_predictions)
        if most_frequent_prediction is not None:
            most_predicted_class = class_map[int(most_frequent_prediction)]
            tracklet_predictions.append(most_predicted_class)
            tracklet_weights.append(total_duration)

    if tracklet_predictions:
        final_weighted_prediction = weighted_prediction(
            tracklet_predictions, tracklet_weights
        )
        return final_weighted_prediction
    else:
        return "UnClassified"


def save_cell_type_predictions(
    tracks_dataframe, cell_map, predictions, save_dir, channel
):

    cell_type = {}
    for value in cell_map.values():
        cell_type[value] = pd.DataFrame(
            columns=["TrackMate Track ID", "t", "z", "y", "x"]
        )
        for k, v in predictions.items():
            if value == v:

                current_track_dataframe = tracks_dataframe[
                    tracks_dataframe["TrackMate Track ID"] == k
                ]
                t_min = current_track_dataframe["t"].idxmin()
                x = current_track_dataframe.loc[t_min, "x"]
                y = current_track_dataframe.loc[t_min, "y"]
                z = current_track_dataframe.loc[t_min, "z"]
                t = current_track_dataframe.loc[t_min, "t"]
                new_row = pd.DataFrame(
                    {"TrackMate Track ID": [k], "t": [t], "z": [z], "y": [y], "x": [x]}
                )
                cell_type[value] = pd.concat([cell_type[value], new_row])

    for value, data in cell_type.items():
        df = pd.DataFrame(data)
        save_name = f"{value}_inception"

        if "Goblet" in value:
            save_name = f"goblet_cells_{channel}annotations_inception"
        if "Radial" in value:
            save_name = f"radially_intercalating_cells_{channel}annotations_inception"
        if "Basal" in value:
            save_name = f"basal_cells_{channel}annotations_inception"

        filename = os.path.join(save_dir, f"{save_name}.csv")
        df.to_csv(filename, index=True)
        print(f"Saved data for cell type {value} to {filename}")

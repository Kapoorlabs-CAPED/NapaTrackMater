from .Trackmate import TrackMate, get_feature_dict
from pathlib import Path
import lxml.etree as et
import concurrent
import os
import numpy as np
import napari
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from typing import List, Union
from torchsummary import summary
import torch.nn.init as init


class TrackVector(TrackMate):
    def __init__(
        self,
        viewer: napari.Viewer,
        image: np.ndarray,
        master_xml_path: Path,
        spot_csv_path: Path,
        track_csv_path: Path,
        edges_csv_path: Path,
        t_minus: int = 0,
        t_plus: int = 10,
        x_start: int = 0,
        x_end: int = 10,
        y_start: int = 0,
        y_end: int = 10,
        show_tracks: bool = True,
        autoencoder_model=None,
        num_points=2048,
        latent_features=1,
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
        self.all_track_properties = []
        self.split_points_times = []

        self.AllTrackIds.append(None)
        self.DividingTrackIds.append(None)
        self.NormalTrackIds.append(None)

        self.AllTrackIds.append(self.TrackidBox)
        self.DividingTrackIds.append(self.TrackidBox)
        self.NormalTrackIds.append(self.TrackidBox)

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
        self.detectorchannel = int(float(self.detectorsettings.get("TARGET_CHANNEL")))
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
                    motion_angle,
                    acceleration,
                    distance_cell_mask,
                    radial_angle,
                    cell_axis_mask,
                    _,
                    _,
                    _,
                    _,
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
                    volume,
                    eccentricity_comp_first,
                    eccentricity_comp_second,
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
                        volume,
                        eccentricity_comp_first,
                        eccentricity_comp_second,
                        surface_area,
                        speed,
                        motion_angle,
                        acceleration,
                        distance_cell_mask,
                        radial_angle,
                        cell_axis_mask,
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

        print("getting attributes")
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
        np.save(save_path + "_counts.npy", df.to_numpy())

        max_number_dividing = full_dataframe["Number_Dividing"].max()
        min_number_dividing = full_dataframe["Number_Dividing"].min()
        excluded_keys = ["Track ID", "t", "z", "y", "x"]
        for i in range(
            min_number_dividing.astype(int), max_number_dividing.astype(int) + 1
        ):
            for column in full_dataframe.columns:
                if column not in excluded_keys:
                    data = full_dataframe[column][
                        full_dataframe["Number_Dividing"].astype(int) == i
                    ]
                    np.save(
                        f"{save_path}_{column}_Number_Dividing_{i}.npy", data.to_numpy()
                    )

        all_split_data = []
        for split_id in self.split_cell_ids:
            spot_properties = self.unique_spot_properties[split_id]
            track_id = spot_properties[self.trackid_key]
            unique_id = spot_properties[self.uniqueid_key]
            tracklet_id = spot_properties[self.trackletid_key]
            number_times_divided = spot_properties[self.number_dividing_key]
            surface_area = spot_properties[self.surface_area_key]
            eccentricity_comp_first = spot_properties[self.eccentricity_comp_firstkey]
            eccentricity_comp_second = spot_properties[self.eccentricity_comp_secondkey]
            volume = spot_properties[self.quality_key]
            radius = spot_properties[self.radius_key]
            speed = spot_properties[self.speed_key]
            motion_angle = spot_properties[self.motion_angle_key]
            acceleration = spot_properties[self.acceleration_key]
            distance_cell_mask = spot_properties[self.distance_cell_mask_key]
            radial_angle = spot_properties[self.radial_angle_key]
            cell_axis_mask = spot_properties[self.cellaxis_mask_key]

            data = {
                "Track ID": track_id,
                "Unique ID": unique_id,
                "Tracklet ID": tracklet_id,
                "Number Times Divided": number_times_divided,
                "Surface Area": surface_area,
                "Eccentricity Comp First": eccentricity_comp_first,
                "Eccentricity Comp Second": eccentricity_comp_second,
                "Volume": volume,
                "Radius": radius,
                "Speed": speed,
                "Motion Angle": motion_angle,
                "Acceleration": acceleration,
                "Distance Cell Mask": distance_cell_mask,
                "Radial Angle": radial_angle,
                "Cell Axis Mask": cell_axis_mask,
            }

            all_split_data.append(data)

        np.save(f"{save_path}_data_at_mitosis_time.npy", all_split_data)

    def get_shape_dynamic_feature_dataframe(self):

        current_shape_dynamic_vectors = self.current_shape_dynamic_vectors
        global_shape_dynamic_dataframe = []

        for i in range(len(current_shape_dynamic_vectors)):
            vector_list = current_shape_dynamic_vectors[i]
            initial_array = np.array(vector_list[:18])
            latent_shape_features = np.array(vector_list[18:])
            zipped_initial_array = list(zip(initial_array))
            data_frame_list = np.transpose(
                np.asarray(
                    [zipped_initial_array[i] for i in range(len(zipped_initial_array))]
                )[:, 0, :]
            )

            shape_dynamic_dataframe = pd.DataFrame(
                data_frame_list,
                columns=[
                    "Track ID",
                    "t",
                    "z",
                    "y",
                    "x",
                    "Dividing",
                    "Number_Dividing",
                    "Radius",
                    "Volume",
                    "Eccentricity Comp First",
                    "Eccentricity Comp Second",
                    "Surface Area",
                    "Speed",
                    "Motion_Angle",
                    "Acceleration",
                    "Distance_Cell_mask",
                    "Radial_Angle",
                    "Cell_Axis_Mask",
                ],
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

            if len(global_shape_dynamic_dataframe) == 0:
                global_shape_dynamic_dataframe = shape_dynamic_dataframe
            else:
                global_shape_dynamic_dataframe = pd.concat(
                    [global_shape_dynamic_dataframe, shape_dynamic_dataframe],
                    ignore_index=True,
                )

        global_shape_dynamic_dataframe = global_shape_dynamic_dataframe.sort_values(
            by=["Track ID"]
        )
        global_shape_dynamic_dataframe = global_shape_dynamic_dataframe.sort_values(
            by=["t"]
        )

        return global_shape_dynamic_dataframe


def create_training_tracklets(
    global_shape_dynamic_dataframe: pd.DataFrame, t_minus=20, t_plus=20
):
    training_tracklets = {}
    subset = global_shape_dynamic_dataframe[
        global_shape_dynamic_dataframe["Dividing"] == 1
    ].loc[
        global_shape_dynamic_dataframe.duplicated(
            subset=["t", "x", "y", "z"], keep=False
        )
    ]
    dividing_track_ids = subset["Track ID"].unique()
    for track_id in dividing_track_ids:
        track_dividing_times = subset[subset["Track ID"] == track_id]["t"].unique()

        for dividing_time in track_dividing_times:
            lower_bound = max(0, dividing_time - t_minus)
            upper_bound = dividing_time + t_plus
            track_data = global_shape_dynamic_dataframe[
                (global_shape_dynamic_dataframe["Track ID"] == track_id)
                & (global_shape_dynamic_dataframe["t"] >= lower_bound)
                & (global_shape_dynamic_dataframe["t"] <= upper_bound)
            ].sort_values(by="t")
            if track_data.shape[0] > 0:
                shape_dynamic_dataframe = track_data[
                    [
                        "Radius",
                        "Volume",
                        "Eccentricity Comp First",
                        "Eccentricity Comp Second",
                        "Surface Area",
                        "Speed",
                        "Motion_Angle",
                        "Acceleration",
                        "Distance_Cell_mask",
                        "Radial_Angle",
                        "Cell_Axis_Mask",
                    ]
                ].copy()

                shape_dataframe = track_data[
                    [
                        "Radius",
                        "Volume",
                        "Eccentricity Comp First",
                        "Eccentricity Comp Second",
                        "Surface Area",
                    ]
                ].copy()

                dynamic_dataframe = track_data[
                    [
                        "Speed",
                        "Motion_Angle",
                        "Acceleration",
                        "Distance_Cell_mask",
                        "Radial_Angle",
                        "Cell_Axis_Mask",
                    ]
                ].copy()

                full_dataframe = track_data[
                    [
                        "Track ID",
                        "t",
                        "z",
                        "y",
                        "x",
                        "Dividing",
                        "Number_Dividing",
                        "Radius",
                        "Volume",
                        "Eccentricity Comp First",
                        "Eccentricity Comp Second",
                        "Surface Area",
                        "Speed",
                        "Motion_Angle",
                        "Acceleration",
                        "Distance_Cell_mask",
                        "Radial_Angle",
                        "Cell_Axis_Mask",
                    ]
                ].copy()

                latent_columns = [
                    col
                    for col in track_data.columns
                    if col.startswith("latent_feature_number_")
                ]
                if latent_columns:
                    latent_features = track_data[latent_columns].copy()
                    full_dataframe = pd.concat(
                        [full_dataframe, latent_features], axis=1
                    )
                    shape_dataframe = pd.concat(
                        [shape_dataframe, latent_features], axis=1
                    )
                    shape_dynamic_dataframe = pd.concat(
                        [shape_dynamic_dataframe, latent_features], axis=1
                    )
                shape_dynamic_dataframe_list = shape_dynamic_dataframe.to_dict(
                    orient="records"
                )
                shape_dataframe_list = shape_dataframe.to_dict(orient="records")
                dynamic_dataframe_list = dynamic_dataframe.to_dict(orient="records")
                full_dataframe_list = full_dataframe.to_dict(orient="records")
                training_tracklets[track_id] = (
                    shape_dynamic_dataframe_list,
                    shape_dataframe_list,
                    dynamic_dataframe_list,
                    full_dataframe_list,
                )
    subset = global_shape_dynamic_dataframe[
        global_shape_dynamic_dataframe["Dividing"] == 0
    ]
    non_dividing_track_ids = subset["Track ID"].unique()
    for track_id in non_dividing_track_ids:
        track_data = subset[(subset["Track ID"] == track_id)].sort_values(by="t")
        track_data = global_shape_dynamic_dataframe[
            (global_shape_dynamic_dataframe["Track ID"] == track_id)
        ].sort_values(by="t")

        if track_data.shape[0] > 0:
            shape_dynamic_dataframe = track_data[
                [
                    "Radius",
                    "Volume",
                    "Eccentricity Comp First",
                    "Eccentricity Comp Second",
                    "Surface Area",
                    "Speed",
                    "Motion_Angle",
                    "Acceleration",
                    "Distance_Cell_mask",
                    "Radial_Angle",
                    "Cell_Axis_Mask",
                ]
            ].copy()

            shape_dataframe = track_data[
                [
                    "Radius",
                    "Volume",
                    "Eccentricity Comp First",
                    "Eccentricity Comp Second",
                    "Surface Area",
                ]
            ].copy()

            dynamic_dataframe = track_data[
                [
                    "Speed",
                    "Motion_Angle",
                    "Acceleration",
                    "Distance_Cell_mask",
                    "Radial_Angle",
                    "Cell_Axis_Mask",
                ]
            ].copy()

            full_dataframe = track_data[
                [
                    "Track ID",
                    "t",
                    "z",
                    "y",
                    "x",
                    "Dividing",
                    "Number_Dividing",
                    "Radius",
                    "Volume",
                    "Eccentricity Comp First",
                    "Eccentricity Comp Second",
                    "Surface Area",
                    "Speed",
                    "Motion_Angle",
                    "Acceleration",
                    "Distance_Cell_mask",
                    "Radial_Angle",
                    "Cell_Axis_Mask",
                ]
            ].copy()

            latent_columns = [
                col
                for col in track_data.columns
                if col.startswith("latent_feature_number_")
            ]
            if latent_columns:
                latent_features = track_data[latent_columns].copy()
                full_dataframe = pd.concat([full_dataframe, latent_features], axis=1)
                shape_dataframe = pd.concat([shape_dataframe, latent_features], axis=1)
                shape_dynamic_dataframe = pd.concat(
                    [shape_dynamic_dataframe, latent_features], axis=1
                )
            shape_dynamic_dataframe_list = shape_dynamic_dataframe.to_dict(
                orient="records"
            )
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


def create_analysis_vectors_dict(global_shape_dynamic_dataframe: pd.DataFrame):
    analysis_vectors = {}
    for track_id in global_shape_dynamic_dataframe["Track ID"].unique():
        track_data = global_shape_dynamic_dataframe[
            global_shape_dynamic_dataframe["Track ID"] == track_id
        ].sort_values(by="t")

        shape_dynamic_dataframe = track_data[
            [
                "Radius",
                "Volume",
                "Eccentricity Comp First",
                "Eccentricity Comp Second",
                "Surface Area",
                "Speed",
                "Motion_Angle",
                "Acceleration",
                "Distance_Cell_mask",
                "Radial_Angle",
                "Cell_Axis_Mask",
            ]
        ].copy()

        shape_dataframe = track_data[
            [
                "Radius",
                "Volume",
                "Eccentricity Comp First",
                "Eccentricity Comp Second",
                "Surface Area",
            ]
        ].copy()

        dynamic_dataframe = track_data[
            [
                "Speed",
                "Motion_Angle",
                "Acceleration",
                "Distance_Cell_mask",
                "Radial_Angle",
                "Cell_Axis_Mask",
            ]
        ].copy()

        full_dataframe = track_data[
            [
                "Track ID",
                "t",
                "z",
                "y",
                "x",
                "Dividing",
                "Number_Dividing",
                "Radius",
                "Volume",
                "Eccentricity Comp First",
                "Eccentricity Comp Second",
                "Surface Area",
                "Speed",
                "Motion_Angle",
                "Acceleration",
                "Distance_Cell_mask",
                "Radial_Angle",
                "Cell_Axis_Mask",
            ]
        ].copy()

        latent_columns = [
            col
            for col in track_data.columns
            if col.startswith("latent_feature_number_")
        ]
        if latent_columns:
            latent_features = track_data[latent_columns].copy()
            full_dataframe = pd.concat([full_dataframe, latent_features], axis=1)
            shape_dataframe = pd.concat([shape_dataframe, latent_features], axis=1)
            shape_dynamic_dataframe = pd.concat(
                [shape_dynamic_dataframe, latent_features], axis=1
            )
        shape_dynamic_dataframe_list = shape_dynamic_dataframe.to_dict(orient="records")
        shape_dataframe_list = shape_dataframe.to_dict(orient="records")
        dynamic_dataframe_list = dynamic_dataframe.to_dict(orient="records")
        full_dataframe_list = full_dataframe.to_dict(orient="records")
        analysis_vectors[track_id] = (
            shape_dynamic_dataframe_list,
            shape_dataframe_list,
            dynamic_dataframe_list,
            full_dataframe_list,
        )

    return analysis_vectors


def z_score_normalization(data):
    normalized_data = data  # normalize(data, norm='l2')
    return normalized_data


def append_data_to_npz(file_path, key, data):
    existing_data = np.load(file_path, allow_pickle=True)[key]

    training_data_key = []
    features_list = []
    label_dividing_list = []
    label_number_dividing_list = []

    for entry in existing_data:
        features = entry["features"]
        label_dividing = entry["label_dividing"]
        label_number_dividing = entry["label_number_dividing"]

        features_list.append(features)
        label_dividing_list.append(label_dividing)
        label_number_dividing_list.append(label_number_dividing)
    for entry in data:
        features = entry["features"]
        label_dividing = entry["label_dividing"]
        label_number_dividing = entry["label_number_dividing"]

        features_list.append(features)
        label_dividing_list.append(label_dividing)
        label_number_dividing_list.append(label_number_dividing)
    training_data_key.append(
        {
            "features": features_list,
            "label_dividing": label_dividing_list,
            "label_number_dividing": label_number_dividing_list,
        }
    )
    return training_data_key


def create_embeddings_with_gt(
    shape_dynamic_track_arrays,
    shape_track_arrays,
    dynamic_track_arrays,
    global_shape_dynamic_dataframe,
    analysis_track_ids,
):
    prediction_data_shape_dynamic = []
    prediction_data_shape = []
    prediction_data_dynamic = []
    analysis_track_ids = np.asarray(analysis_track_ids)
    shape_dynamic_track_arrays = z_score_normalization(shape_dynamic_track_arrays)
    shape_track_arrays = z_score_normalization(shape_track_arrays)
    dynamic_track_arrays = z_score_normalization(dynamic_track_arrays)

    for idx in range(analysis_track_ids.shape[0]):
        current_track_id = analysis_track_ids[idx]
        filtered_data = global_shape_dynamic_dataframe[
            global_shape_dynamic_dataframe["Track ID"] == current_track_id
        ]

        label_dividing = filtered_data["Dividing"].values[0]
        label_number_dividing = filtered_data["Number_Dividing"].values[0]

        gt_label = int(label_dividing)
        gt_label_number = int(label_number_dividing)

        features_shape_dynamic = shape_dynamic_track_arrays[idx, :].tolist()
        features_shape = shape_track_arrays[idx, :].tolist()
        features_dynamic = dynamic_track_arrays[idx, :].tolist()
        prediction_data_shape_dynamic.append(
            {
                "features": features_shape_dynamic,
                "label_dividing": gt_label,
                "label_number_dividing": gt_label_number,
            }
        )

        prediction_data_shape.append(
            {
                "features": features_shape,
                "label_dividing": gt_label,
                "label_number_dividing": gt_label_number,
            }
        )

        prediction_data_dynamic.append(
            {
                "features": features_dynamic,
                "label_dividing": gt_label,
                "label_number_dividing": gt_label_number,
            }
        )

    return (
        np.array(prediction_data_shape_dynamic),
        np.array(prediction_data_shape),
        np.array(prediction_data_dynamic),
    )


def load_prediction_data(prediction_data_feature):
    loaded_data = prediction_data_feature
    features_list = []
    label_dividing_list = []
    label_number_dividing_list = []

    for entry in loaded_data:
        features = entry["features"]
        label_dividing = entry["label_dividing"]
        label_number_dividing = entry["label_number_dividing"]

        features_list.append(features)
        label_dividing_list.append(label_dividing)
        label_number_dividing_list.append(label_number_dividing)

    features_array = np.array(features_list)
    label_dividing_array = np.array(label_dividing_list)
    label_number_dividing_array = np.array(label_number_dividing_list)

    return features_array, label_dividing_array, label_number_dividing_array


def create_mitosis_training_data(
    shape_dynamic_track_arrays,
    shape_track_arrays,
    dynamic_track_arrays,
    global_shape_dynamic_dataframe,
    analysis_track_ids,
    save_path,
    append_data=False,
):
    training_data_shape_dynamic = []
    training_data_shape = []
    training_data_dynamic = []
    analysis_track_ids = np.asarray(analysis_track_ids)
    shape_dynamic_track_arrays = z_score_normalization(shape_dynamic_track_arrays)
    shape_track_arrays = z_score_normalization(shape_track_arrays)
    dynamic_track_arrays = z_score_normalization(dynamic_track_arrays)
    for idx in range(analysis_track_ids.shape[0]):
        current_track_id = analysis_track_ids[idx]
        filtered_data = global_shape_dynamic_dataframe[
            global_shape_dynamic_dataframe["Track ID"] == current_track_id
        ]

        label_dividing = filtered_data["Dividing"].values[0]
        label_number_dividing = filtered_data["Number_Dividing"].values[0]

        gt_label = int(label_dividing)
        gt_label_number = int(label_number_dividing)

    for idx in range(analysis_track_ids.shape[0]):
        current_track_id = analysis_track_ids[idx]
        filtered_data = global_shape_dynamic_dataframe[
            global_shape_dynamic_dataframe["Track ID"] == current_track_id
        ]

        label_dividing = filtered_data["Dividing"].values[0]
        label_number_dividing = filtered_data["Number_Dividing"].values[0]

        gt_label = int(label_dividing)
        gt_label_number = int(label_number_dividing)

        features_shape_dynamic = shape_dynamic_track_arrays[idx, :].tolist()
        features_shape = shape_track_arrays[idx, :].tolist()
        features_dynamic = dynamic_track_arrays[idx, :].tolist()
        if (
            np.any(features_shape_dynamic)
            and np.any(features_shape)
            and np.any(features_dynamic)
        ):

            training_data_shape_dynamic.append(
                {
                    "features": features_shape_dynamic,
                    "label_dividing": gt_label,
                    "label_number_dividing": gt_label_number,
                }
            )

            training_data_shape.append(
                {
                    "features": features_shape,
                    "label_dividing": gt_label,
                    "label_number_dividing": gt_label_number,
                }
            )

            training_data_dynamic.append(
                {
                    "features": features_dynamic,
                    "label_dividing": gt_label,
                    "label_number_dividing": gt_label_number,
                }
            )
    if (
        len(training_data_shape_dynamic) > 0
        and len(training_data_shape) > 0
        and len(training_data_dynamic) > 0
    ):
        if append_data:
            training_data_shape_dynamic = append_data_to_npz(
                os.path.join(save_path, "shape_dynamic.npz"),
                "shape_dynamic",
                training_data_shape_dynamic,
            )
            training_data_shape = append_data_to_npz(
                os.path.join(save_path, "shape.npz"), "shape", training_data_shape
            )
            training_data_dynamic = append_data_to_npz(
                os.path.join(save_path, "dynamic.npz"), "dynamic", training_data_dynamic
            )
        np.savez(
            os.path.join(save_path, "shape_dynamic.npz"),
            shape_dynamic=np.array(training_data_shape_dynamic),
        )
        np.savez(
            os.path.join(save_path, "shape.npz"), shape=np.array(training_data_shape)
        )
        np.savez(
            os.path.join(save_path, "dynamic.npz"),
            dynamic=np.array(training_data_dynamic),
        )

    return training_data_shape_dynamic, training_data_shape, training_data_dynamic


def load_training_data_npz(file_path, key):
    loaded_data = np.load(file_path, allow_pickle=True)[key]
    features_list = []
    label_dividing_list = []
    label_number_dividing_list = []

    for entry in loaded_data:
        features = entry["features"]
        label_dividing = entry["label_dividing"]
        label_number_dividing = entry["label_number_dividing"]

        features_list.append(features)
        label_dividing_list.append(label_dividing)
        label_number_dividing_list.append(label_number_dividing)

    features_array = np.array(features_list)
    label_dividing_array = np.array(label_dividing_list)
    label_number_dividing_array = np.array(label_number_dividing_list)

    return features_array, label_dividing_array, label_number_dividing_array


def load_training_data(save_path):
    shape_dynamic_path = os.path.join(save_path, "shape_dynamic.npz")
    shape_path = os.path.join(save_path, "shape.npz")
    dynamic_path = os.path.join(save_path, "dynamic.npz")

    shape_dynamic_data = load_training_data_npz(shape_dynamic_path, "shape_dynamic")
    shape_data = load_training_data_npz(shape_path, "shape")
    dynamic_data = load_training_data_npz(dynamic_path, "dynamic")

    return shape_dynamic_data, shape_data, dynamic_data


def train_mitosis_classifier(
    features_array,
    labels_array,
    save_path,
    model_type="KNN",
    n_neighbors=5,
    random_state=42,
):
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, labels_array, test_size=0.2, random_state=random_state
    )
    X_train = X_train
    y_train = y_train.astype(np.uint8)

    X_test = X_test
    y_test = y_test.astype(np.uint8)
    if model_type == "KNN":
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        dump(knn, save_path + "knn_mitosis_model.joblib")
        return accuracy
    elif model_type == "RandomForest":
        rf = RandomForestClassifier(random_state=random_state)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test)
        dump(rf, save_path + "random_forest_mitosis_model.joblib")
        return accuracy
    else:
        raise ValueError(
            "Invalid model_type. Choose 'KNN' or 'RandomForest Classifier'."
        )


def create_gt_analysis_vectors_dict(global_shape_dynamic_dataframe: pd.DataFrame):
    gt_analysis_vectors = {}
    for track_id in global_shape_dynamic_dataframe["Track ID"].unique():
        track_data = global_shape_dynamic_dataframe[
            global_shape_dynamic_dataframe["Track ID"] == track_id
        ].sort_values(by="t")
        shape_dynamic_dataframe = track_data[
            [
                "Radius",
                "Volume",
                "Eccentricity Comp First",
                "Eccentricity Comp Second",
                "Surface Area",
                "Speed",
                "Motion_Angle",
                "Acceleration",
                "Distance_Cell_mask",
                "Radial_Angle",
                "Cell_Axis_Mask",
            ]
        ]
        gt_dataframe = track_data[
            [
                "Cluster",
            ]
        ]

        full_dataframe = track_data[
            [
                "Track ID",
                "t",
                "z",
                "y",
                "x",
                "Dividing",
                "Number_Dividing",
                "Radius",
                "Volume",
                "Eccentricity Comp First",
                "Eccentricity Comp Second",
                "Surface Area",
                "Speed",
                "Motion_Angle",
                "Acceleration",
                "Distance_Cell_mask",
                "Radial_Angle",
                "Cell_Axis_Mask",
                "Cluster",
            ]
        ]

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


def supervised_clustering(
    csv_file_name, gt_analysis_vectors, n_neighbors=10, n_estimators=50, method="knn"
):
    csv_file_name_original = csv_file_name + "_training_data"
    data_list = []
    track_ids = []
    for track_id, (
        shape_dynamic_dataframe_list,
        gt_dataframe_list,
        full_dataframe_list,
    ) in gt_analysis_vectors.items():

        shape_dynamic_track_array = np.array(
            [
                [item for item in record.values()]
                for record in shape_dynamic_dataframe_list
            ]
        )

        gt_track_array = np.array(
            [[item for item in record.values()] for record in gt_dataframe_list]
        )
        if shape_dynamic_track_array.shape[0] > 1:
            track_ids.append(track_id)
        if not np.isnan(gt_track_array[0]) and shape_dynamic_track_array.shape[0] > 1:
            (
                shape_dynamic_covariance,
                shape_dynamic_eigenvectors,
            ) = compute_covariance_matrix(shape_dynamic_track_array, feature_array=11)

            upper_triangle_indices = np.triu_indices_from(shape_dynamic_covariance)

            flattened_covariance = shape_dynamic_covariance[upper_triangle_indices]
            data_list.append(
                {
                    "Flattened_Covariance": flattened_covariance,
                    "gt_label": gt_track_array[0][0],
                }
            )
    result_dataframe = pd.DataFrame(data_list)
    if os.path.exists(csv_file_name_original):
        os.remove(csv_file_name_original)
    result_dataframe.to_csv(csv_file_name_original + ".csv", index=False)
    if method == "knn":
        X = np.vstack(result_dataframe["Flattened_Covariance"].values)
        y = result_dataframe["gt_label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        unique_train_labels, count_train_labels = np.unique(y_train, return_counts=True)

        unique_test_labels, count_test_labels = np.unique(y_test, return_counts=True)

        print(f"Training labels: {unique_train_labels}")
        print(f"Training label counts: {count_train_labels}")
        print(f"Testing labels: {unique_test_labels}")
        print(f"Testing label counts: {count_test_labels}")
        print(
            f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}"
        )
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"Model Accuracy on test: {accuracy:.2f}")
        accuracy = model.score(X_train, y_train)
        print(f"Model Accuracy on train: {accuracy:.2f}")

        model_filename = csv_file_name + "_knn_model.joblib"
        dump(model, model_filename)
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=42
        )

        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        accuracy_test = accuracy_score(y_test, y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        print(f"Model Accuracy on test: {accuracy_test:.2f}")
        print(f"Model Accuracy on train: {accuracy_train:.2f}")

        model_filename = "random_forest_model.joblib"
        dump(model, model_filename)

    return model


def predict_supervised_clustering(
    model: KNeighborsClassifier, csv_file_name, full_dataframe, analysis_vectors
):
    track_ids = []
    data_list = []
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
        if shape_dynamic_track_array.shape[0] > 1:
            track_ids.append(track_id)
            (
                shape_dynamic_covariance,
                shape_dynamic_eigenvectors,
            ) = compute_covariance_matrix(shape_dynamic_track_array, feature_array=11)

            upper_triangle_indices = np.triu_indices_from(shape_dynamic_covariance)

            flattened_covariance = shape_dynamic_covariance[upper_triangle_indices]
            data_list.append(
                {
                    "Flattened_Covariance": flattened_covariance,
                }
            )
    result_dataframe = pd.DataFrame(data_list)
    X = np.vstack(result_dataframe["Flattened_Covariance"].values)

    class_labels = model.predict(X)

    track_id_to_cluster = {
        track_id: cluster_label
        for track_id, cluster_label in zip(track_ids, class_labels)
    }
    full_dataframe["Cluster"] = full_dataframe["Track ID"].map(track_id_to_cluster)
    result_dataframe = full_dataframe[["Track ID", "t", "z", "y", "x", "Cluster"]]
    csv_file_name = csv_file_name + ".csv"

    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)
    result_dataframe.to_csv(csv_file_name, index=False)


def calculate_wcss(data, labels, centroids):
    wcss = 0
    label_to_index = {label: i for i, label in enumerate(np.unique(labels))}
    for i in range(len(data)):
        cluster_label = labels[i]
        if cluster_label != -1:
            centroid = centroids[label_to_index[cluster_label]]
            distance = np.linalg.norm(data[i] - centroid)
            wcss += distance**2
    return wcss


def calculate_cluster_centroids(data, labels):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_data = data[labels == label]
            centroid = np.mean(cluster_data, axis=0)
            centroids.append(centroid)
    return np.array(centroids)


def unsupervised_clustering(
    full_dataframe,
    csv_file_name,
    analysis_vectors,
    num_clusters=3,
    metric="euclidean",
    method="ward",
    criterion="maxclust",
):
    csv_file_name_original = csv_file_name
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
            (
                shape_dynamic_covariance,
                shape_dynamic_eigenvectors,
            ) = compute_covariance_matrix(shape_dynamic_track_array, feature_array=11)
            shape_covariance, shape_eigenvectors = compute_covariance_matrix(
                shape_track_array
            )
            dynamic_covaraince, dynamic_eigenvectors = compute_covariance_matrix(
                dynamic_track_array, feature_array=6
            )

            shape_dynamic_covariance_matrix.append(shape_dynamic_covariance)
            shape_covariance_matrix.append(shape_covariance)
            dynamic_covariance_matrix.append(dynamic_covaraince)
            analysis_track_ids.append(track_id)
    shape_dynamic_covariance_3d = np.dstack(shape_dynamic_covariance_matrix)
    shape_covariance_3d = np.dstack(shape_covariance_matrix)
    dynamic_covariance_3d = np.dstack(dynamic_covariance_matrix)

    shape_dynamic_covariance_matrix = np.mean(shape_dynamic_covariance_matrix, axis=0)
    shape_covariance_matrix = np.mean(shape_covariance_matrix, axis=0)
    dynamic_covariance_matrix = np.mean(dynamic_covariance_matrix, axis=0)

    shape_dynamic_covariance_2d = shape_dynamic_covariance_3d.reshape(
        len(analysis_track_ids), -1
    )
    shape_covariance_2d = shape_covariance_3d.reshape(len(analysis_track_ids), -1)
    dynamic_covariance_2d = dynamic_covariance_3d.reshape(len(analysis_track_ids), -1)

    track_arrays_array = [
        shape_dynamic_covariance_matrix,
        shape_covariance_matrix,
        dynamic_covariance_matrix,
    ]

    track_arrays_array_names = ["shape_dynamic", "shape", "dynamic"]
    clusterable_track_arrays = [
        shape_dynamic_covariance_2d,
        shape_covariance_2d,
        dynamic_covariance_2d,
    ]

    for track_arrays in track_arrays_array:
        clusterable_track_array = clusterable_track_arrays[
            track_arrays_array.index(track_arrays)
        ]
        shape_dynamic_cosine_distance = pdist(clusterable_track_array, metric=metric)

        shape_dynamic_linkage_matrix = linkage(
            shape_dynamic_cosine_distance, method=method
        )
        shape_dynamic_cluster_labels = fcluster(
            shape_dynamic_linkage_matrix, num_clusters, criterion=criterion
        )

        cluster_centroids = calculate_cluster_centroids(
            clusterable_track_array, shape_dynamic_cluster_labels
        )
        silhouette = silhouette_score(
            clusterable_track_array, shape_dynamic_cluster_labels, metric=metric
        )
        wcss_value = calculate_wcss(
            clusterable_track_array, shape_dynamic_cluster_labels, cluster_centroids
        )

        silhouette_file_name = os.path.join(
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + f"_silhouette_{metric}_{num_clusters}.npy"
        )
        np.save(silhouette_file_name, silhouette)

        wcss_file_name = os.path.join(
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + f"_wcss_{metric}_{num_clusters}.npy"
        )
        np.save(wcss_file_name, wcss_value)
        track_id_to_cluster = {
            track_id: cluster_label
            for track_id, cluster_label in zip(
                analysis_track_ids, shape_dynamic_cluster_labels
            )
        }
        full_dataframe["Cluster"] = full_dataframe["Track ID"].map(track_id_to_cluster)
        result_dataframe = full_dataframe[["Track ID", "t", "z", "y", "x", "Cluster"]]
        csv_file_name = (
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + ".csv"
        )

        if os.path.exists(csv_file_name):
            os.remove(csv_file_name)
        result_dataframe.to_csv(csv_file_name, index=False)

        mean_matrix_file_name = (
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + f"_{metric}_covariance.npy"
        )
        np.save(mean_matrix_file_name, track_arrays)

        linkage_npy_file_name = (
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + f"_{metric}_linkage.npy"
        )
        np.save(linkage_npy_file_name, shape_dynamic_linkage_matrix)

        cluster_labels_npy_file_name = (
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + f"_{metric}_cluster_labels.npy"
        )
        np.save(cluster_labels_npy_file_name, shape_dynamic_cluster_labels)


def convert_tracks_to_arrays(analysis_vectors, full_dataframe, min_length=None):

    analysis_track_ids = []
    shape_dynamic_covariance_matrix = []
    shape_covariance_matrix = []
    dynamic_covariance_matrix = []
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
        assert (
            shape_dynamic_track_array.shape[0]
            == shape_track_array.shape[0]
            == dynamic_track_array.shape[0]
        ), "Shape dynamic, shape and dynamic track arrays must have the same length."
        if (
            shape_dynamic_track_array.shape[0] > 1
            and shape_dynamic_track_array.shape[0] >= min_length
            if min_length is not None
            else True
        ):
            (
                shape_dynamic_covariance,
                shape_dynamic_eigenvectors,
            ) = compute_covariance_matrix(shape_dynamic_track_array, feature_array=11)
            shape_covariance, shape_eigenvectors = compute_covariance_matrix(
                shape_track_array
            )
            dynamic_covaraince, dynamic_eigenvectors = compute_covariance_matrix(
                dynamic_track_array, feature_array=6
            )
            shape_dynamic_covariance_matrix.append(shape_dynamic_covariance)
            shape_covariance_matrix.append(shape_covariance)
            dynamic_covariance_matrix.append(dynamic_covaraince)
            shape_dynamic_eigenvectors_matrix.extend(shape_dynamic_eigenvectors)
            shape_eigenvectors_matrix.extend(shape_eigenvectors)
            dynamic_eigenvectors_matrix.extend(dynamic_eigenvectors)
            analysis_track_ids.append(track_id)

    # shape_dynamic_covariance_3d = np.dstack(shape_dynamic_covariance_matrix)
    # shape_covariance_3d = np.dstack(shape_covariance_matrix)
    # dynamic_covariance_3d = np.dstack(dynamic_covariance_matrix)

    shape_dynamic_eigenvectors_3d = np.dstack(shape_dynamic_eigenvectors_matrix)
    shape_eigenvectors_3d = np.dstack(shape_eigenvectors_matrix)
    dynamic_eigenvectors_3d = np.dstack(dynamic_eigenvectors_matrix)

    # shape_dynamic_covariance_2d = shape_dynamic_covariance_3d.reshape(
    #    len(analysis_track_ids), -1
    # )
    # shape_covariance_2d = shape_covariance_3d.reshape(len(analysis_track_ids), -1)
    # dynamic_covariance_2d = dynamic_covariance_3d.reshape(len(analysis_track_ids), -1)

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

    # return (
    #    shape_dynamic_covariance_2d,
    #    shape_covariance_2d,
    #    dynamic_covariance_2d,
    #    analysis_track_ids,
    # )

    return (
        shape_dynamic_eigenvectors_1d,
        shape_eigenvectors_1d,
        dynamic_eigenvectors_1d,
        analysis_track_ids,
    )


def compute_covariance_matrix(track_arrays, shape_array=5, feature_array=None):
    for i in range(track_arrays.shape[0]):
        if track_arrays[i, :2].any() == -1:
            if feature_array is not None:
                track_arrays[i, feature_array:] = np.nan
            track_arrays[i, :shape_array] = np.nan

    covariance_matrix = np.cov(track_arrays, rowvar=False)
    covariance_matrix = np.nan_to_num(covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvalue_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalue_order]
    eigenvectors = eigenvectors[:, eigenvalue_order]

    # Split eigenvectors into their real and imaginary parts
    real_part = np.real(eigenvectors)
    imag_part = np.imag(eigenvectors)

    # Concatenate real and imaginary parts along the last axis (axis=1)
    concatenated_eigenvectors = np.concatenate((real_part, imag_part), axis=1)

    return covariance_matrix, concatenated_eigenvectors


class DenseLayer(nn.Module):
    """ """

    def __init__(self, in_channels, growth_rate, bottleneck_size, kernel_size):
        super().__init__()
        self.use_bottleneck = bottleneck_size > 0
        self.num_bottleneck_output_filters = growth_rate * bottleneck_size
        if self.use_bottleneck:
            self.bn2 = nn.GroupNorm(1, in_channels)
            self.act2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(
                in_channels, self.num_bottleneck_output_filters, kernel_size=1, stride=1
            )
        self.bn1 = nn.GroupNorm(1, self.num_bottleneck_output_filters)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            self.num_bottleneck_output_filters,
            growth_rate,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bn2(x)
            x = self.act2(x)
            x = self.conv2(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        return x


class DenseBlock(nn.ModuleDict):
    """ """

    def __init__(
        self, num_layers, in_channels, growth_rate, kernel_size, bottleneck_size
    ):
        super().__init__()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.add_module(
                f"denselayer{i}",
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    bottleneck_size,
                    kernel_size,
                ),
            )

    def forward(self, x):
        layer_outputs = [x]
        for _, layer in self.items():
            x = layer(x)
            layer_outputs.append(x)
            x = torch.cat(layer_outputs, dim=1)
        return x


class TransitionBlock(nn.Module):
    """ """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.GroupNorm(1, in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, dilation=1
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet1d(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 32,
        bottleneck_size: int = 4,
        kernel_size: int = 3,
        in_channels: int = 1,
        num_classes_1: int = 1,
        num_classes_2: int = 1,
    ):

        super().__init__()
        self._initialize_weights()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, num_init_features, kernel_size=3),
            nn.GroupNorm(1, num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
            )
            self.features.add_module(f"denseblock{i}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(
                    in_channels=num_features, out_channels=num_features // 2
                )
                self.features.add_module(f"transition{i}", trans)
                num_features = num_features // 2

        self.final_bn = nn.GroupNorm(1, num_features)
        self.final_act = nn.ReLU(inplace=True)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_1 = nn.Linear(num_features, num_classes_1)
        self.classifier_2 = nn.Linear(num_features, num_classes_2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.GroupNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward_features(self, x):
        out = self.features(x)
        out = self.final_bn(out)
        out = self.final_act(out)
        out = self.final_pool(out)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        features = features.squeeze(-1)
        out_1 = self.classifier_1(features)
        out_2 = self.classifier_2(features)
        return out_1, out_2

    def reset_classifier(self):
        self.classifier = nn.Identity()

    def get_classifier(self):
        return self.classifier


class SimpleDenseNet1d(nn.Module):
    def __init__(
        self, features, num_init_features=32, num_classes_1=1, num_classes_2=1
    ):
        super().__init__()

        # self.features_layer = nn.Sequential(
        #    nn.Conv1d(1, num_init_features, kernel_size=3, padding=1),
        #    nn.BatchNorm1d(num_init_features),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool1d(kernel_size=2)
        # )

        self.input_size = features  # num_init_features #* (features //2)

        self.classifier_1 = nn.Sequential(
            nn.Linear(self.input_size, num_init_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_init_features, num_classes_1),
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(self.input_size, num_init_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_init_features, num_classes_2),
        )

    def forward(self, x):
        # x = self.features_layer(x)
        x = x.view(x.size(0), -1)
        out_1 = self.classifier_1(x)
        out_2 = self.classifier_2(x)
        return out_1, out_2


class MitosisNet(nn.Module):
    def __init__(
        self,
        features,
        growth_rate,
        block_config,
        num_init_features,
        num_classes_class1,
        num_classes_class2,
    ):
        super().__init__()
        self.densenet = DenseNet1d(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            in_channels=1,
            num_classes_1=num_classes_class1,
            num_classes_2=num_classes_class2,
        )
        # SimpleDenseNet1d(
        #    features=features,
        #    num_init_features=num_init_features,
        #    in_channels=1,
        #    num_classes_1=num_classes_class1,
        #    num_classes_2=num_classes_class2,
        # )

        # DenseNet1d(
        #    growth_rate=growth_rate,
        #    block_config=block_config,
        #    num_init_features=num_init_features,
        #    in_channels=1,
        #    num_classes_1=num_classes_class1,
        #    num_classes_2=num_classes_class2,
        # )
        self.num_classes_class1 = num_classes_class1
        self.num_classes_class2 = num_classes_class2

    def forward(self, x):
        class_output1, class_output2 = self.densenet(x)
        return class_output1, class_output2


def train_mitosis_neural_net(
    features_array,
    labels_array_class1,
    labels_array_class2,
    input_size,
    save_path,
    batch_size=64,
    learning_rate=0.001,
    growth_rate: int = 4,
    block_config: tuple = (3, 3),
    num_init_features: int = 32,
    epochs=10,
    use_scheduler=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        X_train,
        X_val,
        y_train_class1,
        y_val_class1,
        y_train_class2,
        y_val_class2,
    ) = train_test_split(
        features_array,
        labels_array_class1.astype(np.uint8),
        labels_array_class2.astype(np.uint8),
        test_size=0.1,
        random_state=42,
    )
    print(
        f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Training labels shape: {y_train_class1.shape}, Validation labels shape: {y_val_class1.shape}"
    )
    X_train_tensor = torch.tensor(X_train).to(device)
    y_train_class1_tensor = torch.tensor(y_train_class1, dtype=torch.uint8).to(device)
    y_train_class2_tensor = torch.tensor(y_train_class2, dtype=torch.uint8).to(device)
    X_val_tensor = torch.tensor(X_val).to(device)
    y_val_class1_tensor = torch.tensor(y_val_class1, dtype=torch.uint8).to(device)
    y_val_class2_tensor = torch.tensor(y_val_class2, dtype=torch.uint8).to(device)

    X_train_tensor = X_train_tensor.float()
    X_val_tensor = X_val_tensor.float()
    num_classes1 = int(torch.max(y_train_class1_tensor)) + 1
    num_classes2 = int(torch.max(y_train_class2_tensor)) + 1
    model_info = {
        "growth_rate": growth_rate,
        "block_config": list(block_config),
        "num_init_features": num_init_features,
        "input_size": input_size,
        "num_classes1": num_classes1,
        "num_classes2": num_classes2,
    }
    with open(save_path + "_model_info.json", "w") as json_file:
        json.dump(model_info, json_file)

    model = MitosisNet(
        features=input_size,
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features,
        num_classes_class1=num_classes1,
        num_classes_class2=num_classes2,
    )

    model.to(device)
    summary(model, (1, input_size))

    criterion_class1 = nn.CrossEntropyLoss()
    criterion_class2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_scheduler:
        milestones = [int(epochs * 0.5), int(epochs * 0.75)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    train_dataset = TensorDataset(
        X_train_tensor, y_train_class1_tensor, y_train_class2_tensor
    )
    val_dataset = TensorDataset(X_val_tensor, y_val_class1_tensor, y_val_class2_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loss_class1_values = []
    train_loss_class2_values = []
    val_loss_class1_values = []
    val_loss_class2_values = []
    train_acc_class1_values = []
    train_acc_class2_values = []
    val_acc_class1_values = []
    val_acc_class2_values = []
    for epoch in range(epochs):
        model.train()
        running_loss_class1 = 0.0
        running_loss_class2 = 0.0
        correct_train_class1 = 0
        total_train_class1 = 0
        correct_train_class2 = 0
        total_train_class2 = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for i, data in enumerate(train_loader):
                inputs, labels_class1, labels_class2 = data
                inputs_with_channel = inputs.unsqueeze(1)
                optimizer.zero_grad()
                class_output1, class_output2 = model(inputs_with_channel)

                loss_class1 = criterion_class1(class_output1, labels_class1)
                loss_class1.backward(retain_graph=True)

                loss_class2 = criterion_class2(class_output2, labels_class2)
                loss_class2.backward()

                optimizer.step()

                outputs_class1, outputs_class2 = model(inputs_with_channel)

                _, predicted_class1 = torch.max(outputs_class1.data, 1)
                _, predicted_class2 = torch.max(outputs_class2.data, 1)

                running_loss_class1 += loss_class1.item()
                running_loss_class2 += loss_class2.item()
                correct_train_class1 += (predicted_class1 == labels_class1).sum().item()
                total_train_class1 += labels_class1.size(0)
                correct_train_class2 += (predicted_class2 == labels_class2).sum().item()
                total_train_class2 += labels_class2.size(0)
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Acc Class1": correct_train_class1 / total_train_class1
                        if total_train_class1 > 0
                        else 0,
                        "Acc Class2": correct_train_class2 / total_train_class2
                        if total_train_class2 > 0
                        else 0,
                        "Class1 Loss": running_loss_class1 / (i + 1),
                        "Class2 Loss": running_loss_class2 / (i + 1),
                    }
                )
            if use_scheduler:
                scheduler.step()
        train_loss_class1_values.append(running_loss_class1 / len(train_loader))
        train_loss_class2_values.append(running_loss_class2 / len(train_loader))
        train_acc_class1_values.append(
            correct_train_class1 / total_train_class1 if total_train_class1 > 0 else 0
        )
        train_acc_class2_values.append(
            correct_train_class2 / total_train_class2 if total_train_class2 > 0 else 0
        )

        model.eval()
        running_val_loss_class1 = 0.0
        running_val_loss_class2 = 0.0
        correct_val_class1 = 0
        total_val_class1 = 0
        correct_val_class2 = 0
        total_val_class2 = 0

        with tqdm(
            total=len(val_loader), desc=f"Validation Epoch {epoch + 1}/{epochs}"
        ) as pbar_val:
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels_class1, labels_class2 = data
                    inputs_with_channel = inputs.unsqueeze(1)
                    outputs_class1, outputs_class2 = model(inputs_with_channel)

                    _, predicted_class1 = torch.max(outputs_class1.data, 1)
                    _, predicted_class2 = torch.max(outputs_class2.data, 1)

                    total_val_class1 += labels_class1.size(0)
                    correct_val_class1 += (
                        (predicted_class1 == labels_class1).sum().item()
                    )

                    total_val_class2 += labels_class2.size(0)
                    correct_val_class2 += (
                        (predicted_class2 == labels_class2).sum().item()
                    )

                    pbar_val.update(1)
                    accuracy_class1 = (
                        correct_val_class1 / total_val_class1
                        if total_val_class1 > 0
                        else 0
                    )
                    accuracy_class2 = (
                        correct_val_class2 / total_val_class2
                        if total_val_class2 > 0
                        else 0
                    )
                    loss_class1 = criterion_class1(outputs_class1, labels_class1)
                    loss_class2 = criterion_class2(outputs_class2, labels_class2)

                    running_val_loss_class1 += loss_class1.item()
                    running_val_loss_class2 += loss_class2.item()
                    pbar_val.set_postfix(
                        {
                            "Acc Class1": accuracy_class1,
                            "Acc Class2": accuracy_class2,
                            "Class1 Loss": running_val_loss_class1 / (i + 1),
                            "Class2 Loss": running_val_loss_class2 / (i + 1),
                        }
                    )

        val_loss_class1_values.append(running_val_loss_class1 / len(val_loader))
        val_loss_class2_values.append(running_val_loss_class2 / len(val_loader))
        val_acc_class1_values.append(
            correct_val_class1 / total_val_class1 if total_val_class1 > 0 else 0
        )
        val_acc_class2_values.append(
            correct_val_class2 / total_val_class2 if total_val_class2 > 0 else 0
        )

    np.savez(
        save_path + "_metrics.npz",
        train_loss_class1=train_loss_class1_values,
        train_loss_class2=train_loss_class2_values,
        val_loss_class1=val_loss_class1_values,
        val_loss_class2=val_loss_class2_values,
        train_acc_class1=train_acc_class1_values,
        train_acc_class2=train_acc_class2_values,
        val_acc_class1=val_acc_class1_values,
        val_acc_class2=val_acc_class2_values,
    )
    torch.save(model.state_dict(), save_path + "_mitosis_track_model.pth")


def plot_metrics_from_npz(npz_file):
    data = np.load(npz_file)

    train_loss_class1 = data["train_loss_class1"]
    train_loss_class2 = data["train_loss_class2"]
    val_loss_class1 = data["val_loss_class1"]
    val_loss_class2 = data["val_loss_class2"]
    train_acc_class1 = data["train_acc_class1"]
    train_acc_class2 = data["train_acc_class2"]
    val_acc_class1 = data["val_acc_class1"]
    val_acc_class2 = data["val_acc_class2"]

    epochs = len(train_loss_class1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss_class1, label="Train Loss Class 1")
    plt.plot(range(epochs), val_loss_class1, label="Validation Loss Class 1")
    plt.legend(loc="lower right")
    plt.title("Loss for Class 1")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_loss_class2, label="Train Loss Class 2")
    plt.plot(range(epochs), val_loss_class2, label="Validation Loss Class 2")
    plt.legend(loc="lower right")
    plt.title("Loss for Class 2")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_acc_class1, label="Train Acc Class 1")
    plt.plot(range(epochs), val_acc_class1, label="Validation Acc Class 1")
    plt.legend(loc="lower right")
    plt.title("Accuracy for Class 1")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_acc_class2, label="Train Acc Class 2")
    plt.plot(range(epochs), val_acc_class2, label="Validation Acc Class 2")
    plt.legend(loc="lower right")
    plt.title("Accuracy for Class 2")

    plt.tight_layout()
    plt.show()


# Model prediction
def predict_with_model(
    saved_model_path, saved_model_json, features_array, threshold=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(saved_model_json) as json_file:
        model_info = json.load(json_file)

    num_classes_class1 = model_info["num_classes1"]
    num_classes_class2 = model_info["num_classes2"]
    growth_rate = model_info["growth_rate"]
    block_config = model_info["block_config"]
    num_init_features = model_info["num_init_features"]
    input_size = model_info["input_size"]

    model = MitosisNet(
        features=input_size,
        growth_rate=growth_rate,
        block_config=tuple(block_config),
        num_init_features=num_init_features,
        num_classes_class1=num_classes_class1,
        num_classes_class2=num_classes_class2,
    )
    model.load_state_dict(
        torch.load(saved_model_path, map_location=torch.device(device))
    )
    model.to(device)
    model.eval()
    predicted_classes1 = []
    predicted_classes2 = []
    for idx in range(features_array.shape[0]):
        feature_type = features_array[idx, :].tolist()
        feature_type = z_score_normalization(feature_type)
        features_tensor = torch.tensor(feature_type).to(device)
        new_data_with_channel = features_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs_class1, outputs_class2 = model(new_data_with_channel)
            predicted_probs_class1 = torch.softmax(outputs_class1, dim=1)
            predicted_probs_class2 = torch.softmax(outputs_class2, dim=1)
            if threshold is not None:
                predicted_probs_class1_numpy = (
                    predicted_probs_class1[:, 1].cpu().detach().numpy()
                )
                predicted_class1 = (predicted_probs_class1_numpy > threshold).astype(
                    int
                )
            else:
                predicted_class1 = (
                    torch.argmax(predicted_probs_class1, dim=1).cpu().numpy()
                )
            predicted_class2 = torch.argmax(predicted_probs_class2, dim=1).cpu().numpy()
            predicted_class1[predicted_class2 == 0] = 0
            predicted_classes1.append(predicted_class1[0])
            predicted_classes2.append(predicted_class2[0])
    return predicted_classes1, predicted_classes2

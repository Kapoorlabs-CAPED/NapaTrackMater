from .Trackmate import TrackMate, get_feature_dict
from pathlib import Path
import lxml.etree as et
import concurrent
import os
import numpy as np
import napari
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import csv
from sklearn.metrics import pairwise_distances
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt


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
    ):

        super().__init__(
            None,
            spot_csv_path,
            track_csv_path,
            edges_csv_path,
            image=image,
            AttributeBoxname="AttributeIDBox",
            TrackAttributeBoxname="TrackAttributeIDBox",
            TrackidBox="All",
            axes="TZYX",
            master_xml_path=None,
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
        xml_parser = et.XMLParser(huge_tree=True)

        self.unique_morphology_dynamic_properties = {}
        self.unique_mitosis_label = {}
        self.non_unique_mitosis_label = {}
        if not isinstance(self.master_xml_path, str):
            if self.master_xml_path.is_file():
                print("Reading Master XML")

                self.xml_content = et.fromstring(
                    open(self.master_xml_path).read().encode(), xml_parser
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
                        executor.submit(self._master_track_computer, track, track_id)
                    )

            [r.result() for r in concurrent.futures.as_completed(futures)]

        print("getting attributes")
        self._get_attributes()

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
                )

        print(
            f"returning shape and dynamic vectors as list {len(self.current_shape_dynamic_vectors)}"
        )

    def _interactive_function(self):

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
        time_counter = Counter(self.cell_id_times)
        times = list(time_counter.keys())
        counts = list(time_counter.values())
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

    def get_shape_dynamic_feature_dataframe(self):

        current_shape_dynamic_vectors = self.current_shape_dynamic_vectors
        global_shape_dynamic_dataframe = []

        for i in range(len(current_shape_dynamic_vectors)):
            vector_list = list(zip(current_shape_dynamic_vectors[i]))
            data_frame_list = np.transpose(
                np.asarray([vector_list[i] for i in range(len(vector_list))])[:, 0, :]
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
        ]
        shape_dataframe = track_data[
            [
                "Radius",
                "Volume",
                "Eccentricity Comp First",
                "Eccentricity Comp Second",
                "Surface Area",
            ]
        ]
        dynamic_dataframe = track_data[
            [
                "Speed",
                "Motion_Angle",
                "Acceleration",
                "Distance_Cell_mask",
                "Radial_Angle",
                "Cell_Axis_Mask",
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
            ]
        ]

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


def create_mitosis_training_data(
    shape_dynamic_track_arrays, shape_track_arrays, dynamic_track_arrays, full_records
):
    training_data_shape_dynamic = []
    training_data_shape = []
    training_data_dynamic = []

    for idx in range(shape_dynamic_track_arrays.shape[0]):
        label_dividing = full_records["Dividing"][idx]
        label_number_dividing = full_records["Number_Dividing"][idx]

        features_shape_dynamic = shape_dynamic_track_arrays[idx, :].tolist()
        features_shape = shape_track_arrays[idx, :].tolist()
        features_dynamic = dynamic_track_arrays[idx, :].tolist()

        # Appending to respective training datasets
        training_data_shape_dynamic.append(
            (features_shape_dynamic, label_dividing, label_number_dividing)
        )

        training_data_shape.append(
            (features_shape, label_dividing, label_number_dividing)
        )

        training_data_dynamic.append(
            (features_dynamic, label_dividing, label_number_dividing)
        )

    return training_data_shape_dynamic, training_data_shape, training_data_dynamic


def extract_neural_training_data(training_data):
    features_list = []
    labels_dividing_list = []
    labels_number_dividing_list = []

    for data_point in training_data:
        features = data_point[0]
        label_dividing = data_point[1]
        label_number_dividing = data_point[2]

        features_list.append(features)
        labels_dividing_list.append(label_dividing)
        labels_number_dividing_list.append(label_number_dividing)

    features_array = np.array(features_list)
    labels_dividing_array = np.array(labels_dividing_list)
    labels_number_dividing_array = np.array(labels_number_dividing_list)

    return features_array, labels_dividing_array, labels_number_dividing_array


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
            ) = compute_covariance_matrix(shape_dynamic_track_array)

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

        # Fit the Random Forest Classifier to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Calculate and print the model accuracy on the test and training sets
        accuracy_test = accuracy_score(y_test, y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        print(f"Model Accuracy on test: {accuracy_test:.2f}")
        print(f"Model Accuracy on train: {accuracy_train:.2f}")

        # Save the trained model to a file
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
            ) = compute_covariance_matrix(shape_dynamic_track_array)

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
    threshold_distance=5.0,
    num_clusters=None,
    metric="euclidean",
    method="ward",
    criterion="distance",
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
            ) = compute_covariance_matrix(shape_dynamic_track_array)
            shape_covariance, shape_eigenvectors = compute_covariance_matrix(
                shape_track_array
            )
            dynamic_covaraince, dynamic_eigenvectors = compute_covariance_matrix(
                dynamic_track_array
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
        if (
            np.isnan(shape_dynamic_cosine_distance).any()
            or np.isinf(shape_dynamic_cosine_distance).any()
        ):
            print(
                "Cosine distance matrix contains NaN or infinite values. Returning an empty linkage matrix."
            )
            return

        shape_dynamic_linkage_matrix = linkage(
            shape_dynamic_cosine_distance, method=method
        )
        if num_clusters is None:
            shape_dynamic_cluster_labels = fcluster(
                shape_dynamic_linkage_matrix, t=threshold_distance, criterion=criterion
            )
        else:
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
            + f"_silhouette_{threshold_distance}.npy"
        )
        np.save(silhouette_file_name, silhouette)

        wcss_file_name = os.path.join(
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + f"_wcss_{threshold_distance}.npy"
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


def convert_tracks_to_arrays(analysis_vectors, full_dataframe):

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
            ) = compute_covariance_matrix(shape_dynamic_track_array)
            shape_covariance, shape_eigenvectors = compute_covariance_matrix(
                shape_track_array
            )
            dynamic_covaraince, dynamic_eigenvectors = compute_covariance_matrix(
                dynamic_track_array
            )

            shape_dynamic_covariance_matrix.append(shape_dynamic_covariance)
            shape_covariance_matrix.append(shape_covariance)
            dynamic_covariance_matrix.append(dynamic_covaraince)
            analysis_track_ids.append(track_id)

    shape_dynamic_covariance_3d = np.dstack(shape_dynamic_covariance_matrix)
    shape_covariance_3d = np.dstack(shape_covariance_matrix)
    dynamic_covariance_3d = np.dstack(dynamic_covariance_matrix)

    shape_dynamic_covariance_2d = shape_dynamic_covariance_3d.reshape(
        len(analysis_track_ids), -1
    )
    shape_covariance_2d = shape_covariance_3d.reshape(len(analysis_track_ids), -1)
    dynamic_covariance_2d = dynamic_covariance_3d.reshape(len(analysis_track_ids), -1)
    return (shape_dynamic_covariance_2d, shape_covariance_2d, dynamic_covariance_2d)


def compute_covariance_matrix(track_arrays):

    covariance_matrix = np.cov(track_arrays, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvalue_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalue_order]
    eigenvectors = eigenvectors[:, eigenvalue_order]

    return covariance_matrix, eigenvectors


class MitosisNet(nn.Module):
    def __init__(self, input_size, num_classes_class1, num_classes_class2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_output_size = self._calculate_conv_output_size(input_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2_class1 = nn.Linear(128, num_classes_class1)
        self.fc3_class2 = nn.Linear(128, num_classes_class2)

    def _calculate_conv_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        class_output1 = torch.softmax(self.fc2_class1(x), dim=1)
        class_output2 = torch.softmax(self.fc3_class2(x), dim=1)
        return class_output1, class_output2


def train_mitosis_neural_net(
    features_array,
    labels_array_class1,
    labels_array_class2,
    input_size,
    save_path,
    batch_size=64,
    learning_rate=0.001,
    epochs=10,
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
        features_array.astype(np.float32),
        labels_array_class1.astype(np.uint8),
        labels_array_class2.astype(np.uint8),
        test_size=0.1,
        random_state=42,
    )
    print(
        f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Training labels shape: {y_train_class1.shape}, Validation labels shape: {y_val_class1.shape}"
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_class1_tensor = torch.tensor(y_train_class1, dtype=torch.uint8).to(device)
    y_train_class2_tensor = torch.tensor(y_train_class2, dtype=torch.uint8).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_class1_tensor = torch.tensor(y_val_class1, dtype=torch.uint8).to(device)
    y_val_class2_tensor = torch.tensor(y_val_class2, dtype=torch.uint8).to(device)

    num_classes1 = int(torch.max(y_train_class1_tensor)) + 1
    num_classes2 = int(torch.max(y_train_class2_tensor)) + 1
    print(f"classes1: {num_classes1}, classes2: {num_classes2}")
    model_info = {
        "input_size": input_size,
        "num_classes1": num_classes1,
        "num_classes2": num_classes2,
    }
    with open(save_path + "_model_info.json", "w") as json_file:
        json.dump(model_info, json_file)
    model = MitosisNet(
        input_size=input_size,
        num_classes_class1=num_classes1,
        num_classes_class2=num_classes2,
    )
    model.to(device)

    criterion_class1 = nn.CrossEntropyLoss()
    criterion_class2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
                optimizer.zero_grad()
                class_output1, class_output2 = model(inputs)

                loss_class1 = criterion_class1(class_output1, labels_class1)
                loss_class1.backward(retain_graph=True)

                loss_class2 = criterion_class2(class_output2, labels_class2)
                loss_class2.backward()

                optimizer.step()

                outputs_class1, outputs_class2 = model(inputs)

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
                    outputs_class1, outputs_class2 = model(inputs)

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
                    pbar_val.set_postfix(
                        {"Acc Class1": accuracy_class1, "Acc Class2": accuracy_class2}
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
    plt.legend()
    plt.title("Loss for Class 1")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_loss_class2, label="Train Loss Class 2")
    plt.plot(range(epochs), val_loss_class2, label="Validation Loss Class 2")
    plt.legend()
    plt.title("Loss for Class 2")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_acc_class1, label="Train Acc Class 1")
    plt.plot(range(epochs), val_acc_class1, label="Validation Acc Class 1")
    plt.legend()
    plt.title("Accuracy for Class 1")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_acc_class2, label="Train Acc Class 2")
    plt.plot(range(epochs), val_acc_class2, label="Validation Acc Class 2")
    plt.legend()
    plt.title("Accuracy for Class 2")

    plt.tight_layout()
    plt.show()


def predict_with_model(saved_model_path, features_array):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(saved_model_path + "_model_info.json") as json_file:
        model_info = json.load(json_file)

    input_size = model_info["input_size"]
    num_classes_class1 = model_info["num_classes1"]
    num_classes_class2 = model_info["num_classes2"]

    model = MitosisNet(
        input_size=input_size,
        num_classes_class1=num_classes_class1,
        num_classes_class2=num_classes_class2,
    )
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    model.eval()

    features_tensor = torch.tensor(features_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs_class1, outputs_class2 = model(features_tensor)

        _, predicted_class1 = torch.max(outputs_class1.data, 1)
        _, predicted_class2 = torch.max(outputs_class2.data, 1)

    predicted_class1 = predicted_class1.cpu().numpy()
    predicted_class2 = predicted_class2.cpu().numpy()

    return predicted_class1, predicted_class2


def _save_feature_importance(
    sorted_feature_names,
    normalized_importances,
    csv_file_name_original,
    track_arrays_array_names,
    track_arrays_array,
    track_arrays,
):
    data = list(zip(sorted_feature_names, normalized_importances))
    csv_file_name = (
        csv_file_name_original
        + track_arrays_array_names[track_arrays_array.index(track_arrays)]
        + "_feature_importance"
        + ".csv"
    )
    with open(csv_file_name, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Feature", "Importance"])
        for feature, importance in data:
            writer.writerow([feature, importance])


def _perform_pca_clustering(track_arrays, num_clusters, num_components=3):
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(track_arrays)

    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(reduced_data)

    return cluster_labels, pca.components_


def _perform_agg_clustering(track_arrays, num_clusters):

    distance_matrix = pairwise_distances(track_arrays, metric="euclidean")
    model = AgglomerativeClustering(
        affinity="precomputed", n_clusters=num_clusters, linkage="ward"
    ).fit(distance_matrix)

    clusters = model.labels_

    return clusters

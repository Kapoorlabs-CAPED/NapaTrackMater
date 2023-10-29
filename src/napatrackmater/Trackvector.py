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
    csv_file_name,
    gt_analysis_vectors,
    num_clusters,
):
    csv_file_name_original = csv_file_name + "_training_data"
    data_list = []

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
        if not np.isnan(gt_track_array[0]) and shape_dynamic_track_array.shape[0] > 1:
            (
                shape_dynamic_covariance,
                shape_dynamic_eigenvectors,
            ) = compute_covariance_matrix(shape_dynamic_track_array)

            flattened_covariance = shape_dynamic_covariance.flatten()
            data_list.append(
                {
                    "Flattened_Covariance": flattened_covariance,
                    "gt_label": gt_track_array[0][0],
                }
            )
    result_dataframe = pd.DataFrame(data_list)
    if os.path.exists(csv_file_name_original):
        os.remove(csv_file_name_original)
    result_dataframe.to_csv(csv_file_name_original, index=False)
    X = np.vstack(result_dataframe["Flattened_Covariance"].values)
    y = result_dataframe["gt_label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=num_clusters)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    model_filename = "knn_model.joblib"
    dump(knn, model_filename)

    return knn


def unsupervised_clustering(
    full_dataframe,
    csv_file_name,
    analysis_vectors,
    num_clusters,
    metric="cosine",
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
        ), "Shape dynamic, shape and dynamic track arrays must have the same length"
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
        shape_dynamic_linkage_matrix = linkage(
            shape_dynamic_cosine_distance, method=method
        )
        shape_dynamic_cluster_labels = fcluster(
            shape_dynamic_linkage_matrix, num_clusters, criterion=criterion
        )
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
            + "_covariance.npy"
        )
        np.save(mean_matrix_file_name, track_arrays)

        linkage_npy_file_name = (
            csv_file_name_original
            + track_arrays_array_names[track_arrays_array.index(track_arrays)]
            + "_linkage.npy"
        )
        np.save(linkage_npy_file_name, shape_dynamic_linkage_matrix)


def convert_tracks_to_arrays(analysis_vectors, min_track_length=0):

    filtered_track_ids = []
    shape_dynamic_track_arrays = []
    shape_track_arrays = []
    dynamic_track_arrays = []
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
        ), "Shape dynamic, shape and dynamic track arrays must have the same length"
        if shape_dynamic_track_array.shape[0] > min_track_length:
            shape_dynamic_track_arrays.append(
                shape_dynamic_track_array.astype(np.float32)
            )
            shape_track_arrays.append(shape_track_array.astype(np.float32))
            dynamic_track_arrays.append(dynamic_track_array.astype(np.float32))
            filtered_track_ids.append(track_id)
    shape_dynamic_track_arrays_array = np.vstack(shape_dynamic_track_arrays)
    shape_track_arrays_array = np.vstack(shape_track_arrays)
    dynamic_track_arrays_array = np.vstack(dynamic_track_arrays)

    return (
        shape_dynamic_track_arrays_array,
        shape_track_arrays_array,
        dynamic_track_arrays_array,
        filtered_track_ids,
    )


def compute_covariance_matrix(track_arrays):

    covariance_matrix = np.cov(track_arrays, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvalue_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalue_order]
    eigenvectors = eigenvectors[:, eigenvalue_order]

    return covariance_matrix, eigenvectors


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

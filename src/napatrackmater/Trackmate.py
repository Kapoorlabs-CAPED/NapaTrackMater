from tqdm import tqdm
import numpy as np
import lxml.etree as et
import csv
from tifffile import imread

import pandas as pd
import math
from skimage import measure
from skimage.segmentation import find_boundaries
from scipy import spatial
from typing import List, Union
from scipy.fftpack import fft, fftfreq
import os
from pathlib import Path
import concurrent
from .clustering import Clustering
from lightning import Trainer


class TrackMate:
    def __init__(
        self,
        xml_path,
        spot_csv_path=None,
        track_csv_path=None,
        edges_csv_path=None,
        AttributeBoxname="AttributeIDBox",
        TrackAttributeBoxname="TrackAttributeIDBox",
        TrackidBox="All",
        second_channel_name="membrane",
        axes="TZYX",
        scale_z=1.0,
        scale_xy=1.0,
        latent_features=1,
        center=True,
        progress_bar=None,
        accelerator: str = "cuda",
        devices: Union[List[int], str, int] = 1,
        master_xml_path: Path = None,
        channel_xml_path: Path = None,
        master_extra_name="",
        seg_image: np.ndarray = None,
        channel_seg_image: np.ndarray = None,
        image: np.ndarray = None,
        mask: np.ndarray = None,
        fourier=True,
        autoencoder_model=None,
        enhance_trackmate_xml: bool = False,
        num_points=2048,
        batch_size=1,
        compute_with_autoencoder=False,
        variable_t_calibration: dict = None,
        oneat_csv_file: str = None,
        goblet_csv_file: str = None,
        basal_csv_file: str = None,
        radial_csv_file: str = None,
        oneat_threshold_cutoff: int = 0.5,
        time_veto: int = 0,
        space_veto: int = 15,
        basal_label: int = 1,
        goblet_label: int = 3,
        radial_label: int = 2,
        enhanced_computation: bool = True
    ):

        self.xml_path = xml_path
        self.master_xml_path = master_xml_path
        self.channel_xml_path = channel_xml_path
        self.spot_csv_path = spot_csv_path
        self.track_csv_path = track_csv_path
        self.edges_csv_path = edges_csv_path
        self.accelerator = accelerator
        self.devices = devices
        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self.center = center
        self.compute_with_autoencoder = compute_with_autoencoder
        self.variable_t_calibration = variable_t_calibration
        self.oneat_csv_file = oneat_csv_file
        self.goblet_csv_file = goblet_csv_file
        self.basal_csv_file = basal_csv_file
        self.radial_csv_file = radial_csv_file
        self.oneat_threshold_cutoff = oneat_threshold_cutoff
        self.latent_features = latent_features
        self.time_veto = time_veto
        self.space_veto = space_veto
        self.basal_label = basal_label
        self.goblet_label = goblet_label
        self.radial_label = radial_label
        self.enhanced_computation = enhanced_computation
        if image is not None:
            self.image = image.astype(np.uint8)
        else:
            self.image = image
        if mask is not None:
            self.mask = mask.astype(np.uint8)
        else:
            self.mask = mask

        self.fourier = fourier
        self.autoencoder_model = autoencoder_model

        if self.autoencoder_model is not None:
            self.pretrainer = Trainer(
                accelerator=self.accelerator, devices=self.devices
            )
        else:
            self.pretrainer = None
        self.enhance_trackmate_xml = enhance_trackmate_xml
        if channel_seg_image is not None:
            self.channel_seg_image = channel_seg_image.astype(np.uint16)
        else:
            self.channel_seg_image = channel_seg_image
        if seg_image is not None:
            self.seg_image = seg_image.astype(np.uint16)
            
        else:
            self.seg_image = seg_image

        self.AttributeBoxname = AttributeBoxname
        self.TrackAttributeBoxname = TrackAttributeBoxname
        self.TrackidBox = TrackidBox
        self.second_channel_name = second_channel_name
        self.master_extra_name = master_extra_name

        self.num_points = num_points
        if self.spot_csv_path is not None:
            self.spot_dataset, self.spot_dataset_index = get_csv_data(
                self.spot_csv_path
            )
        if self.track_csv_path is not None:
            self.track_dataset, self.track_dataset_index = get_csv_data(
                self.track_csv_path
            )
        if self.edges_csv_path is not None:
            self.edges_dataset, self.edges_dataset_index = get_csv_data(
                self.edges_csv_path
            )
        self.progress_bar = progress_bar
        self.axes = axes
        self.batch_size = batch_size

        self.cell_id_times = []
        self.split_cell_ids = []

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
            mean_intensity="MEAN_INTENSITY",
            total_intensity="TOTAL_INTENSITY",
        )
        self.track_analysis_edges_keys = dict(
            spot_source_id="SPOT_SOURCE_ID",
            spot_target_id="SPOT_TARGET_ID",
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
        )

        self.frameid_key = self.track_analysis_spot_keys["frame"]
        self.zposid_key = self.track_analysis_spot_keys["posiz"]
        self.yposid_key = self.track_analysis_spot_keys["posiy"]
        self.xposid_key = self.track_analysis_spot_keys["posix"]
        self.spotid_key = self.track_analysis_spot_keys["spot_id"]
        self.trackid_key = self.track_analysis_spot_keys["track_id"]
        self.radius_key = self.track_analysis_spot_keys["radius"]
        self.quality_key = self.track_analysis_spot_keys["quality"]

        self.generationid_key = "generation_id"
        self.trackletid_key = "tracklet_id"
        self.uniqueid_key = "unique_id"
        self.afterid_key = "after_id"
        self.beforeid_key = "before_id"
        self.rootid_key = "root_id"
        self.msd_key = "msd"
        self.dividing_key = "dividing_normal"
        self.fate_key = "fate"
        self.number_dividing_key = "number_dividing"
        self.distance_cell_mask_key = "distance_cell_mask"
        self.maskcentroid_x_key = "maskcentroid_x_key"
        self.maskcentroid_z_key = "maskcentroid_z_key"
        self.maskcentroid_y_key = "maskcentroid_y_key"
        self.cell_axis_z_key = "cell_axis_z_key"
        self.cell_axis_y_key = "cell_axis_y_key"
        self.cell_axis_x_key = "cell_axis_x_key"
        self.cellid_key = "cell_id"
        self.acceleration_key = "acceleration"
        self.centroid_key = "centroid"
        self.eccentricity_comp_firstkey = "cloud_eccentricity_comp_first"
        self.eccentricity_comp_secondkey = "cloud_eccentricity_comp_second"
        self.eccentricity_comp_thirdkey = "cloud_eccentricity_comp_third"
        self.surface_area_key = "cloud_surfacearea"
        self.radial_angle_z_key = "radial_angle_z_key"
        self.radial_angle_y_key = "radial_angle_y_key"
        self.radial_angle_x_key = "radial_angle_x_key"
        self.motion_angle_z_key = "motion_angle_z"
        self.motion_angle_y_key = "motion_angle_y"
        self.motion_angle_x_key = "motion_angle_x"
        self.local_cell_density_key = "local_density"
        self.latent_shape_features_key = "latent_shape_features"

        self.mean_intensity_ch1_key = self.track_analysis_spot_keys[
            "mean_intensity_ch1"
        ]
        self.mean_intensity_ch2_key = self.track_analysis_spot_keys[
            "mean_intensity_ch2"
        ]
        self.total_intensity_ch1_key = self.track_analysis_spot_keys[
            "total_intensity_ch1"
        ]
        self.total_intensity_ch2_key = self.track_analysis_spot_keys[
            "total_intensity_ch2"
        ]

        self.mean_intensity_key = self.track_analysis_spot_keys["mean_intensity"]
        self.total_intensity_key = self.track_analysis_spot_keys["total_intensity"]

        self.spot_source_id_key = self.track_analysis_edges_keys["spot_source_id"]
        self.spot_target_id_key = self.track_analysis_edges_keys["spot_target_id"]

        self.speed_key = self.track_analysis_edges_keys["speed"]
        self.displacement_key = self.track_analysis_track_keys["track_displacement"]
        self.total_track_distance_key = self.track_analysis_track_keys[
            "total_track_distance"
        ]
        self.max_distance_traveled_key = self.track_analysis_track_keys[
            "max_track_distance"
        ]
        self.track_duration_key = self.track_analysis_track_keys["track_duration"]

        self.edge_time_key = self.track_analysis_edges_keys["edge_time"]
        self.edge_x_location_key = self.track_analysis_edges_keys["edge_x_location"]
        self.edge_y_location_key = self.track_analysis_edges_keys["edge_y_location"]
        self.edge_z_location_key = self.track_analysis_edges_keys["edge_z_location"]

        self.unique_tracks = {}
        self.tracklet_id_to_trackmate_id = {}
        self.tracklet_id_to_generation_id = {}
        self.tracklet_id_to_tracklet_number_id = {}
        self.unique_track_mitosis_label = {}
        self.unique_track_fate_label = {}
        self.unique_track_properties = {}
        self.unique_fft_properties = {}
        self.unique_cluster_properties = {}
        self.unique_shape_properties = {}
        self.unique_dynamic_properties = {}
        self.unique_spot_properties = {}
        self.unique_spot_centroid = {}
        self.unique_oneat_spot_centroid = {}
        self.unique_track_centroid = {}
        self.matched_second_channel_tracks = []
        self.root_spots = {}
        self.all_current_cell_ids = {}
        self.channel_unique_spot_properties = {}
        self.edge_target_lookup = {}
        self.edge_source_lookup = {}
        self.generation_dict = {}
        self.tracklet_dict = {}
        self.graph_split = {}
        self.graph_tracks = {}
        self._timed_centroid = {}
        self.oneat_dividing_tracks = {}
        self.count = 0
        self.cell_veto_box = 0
        xml_parser = et.XMLParser(huge_tree=True)
        if self.master_xml_path is None:
            self.master_xml_path = Path(".")

        if self.master_xml_path.is_dir() and self.xml_path is not None:
            self.xml_content = et.fromstring(
                open(self.xml_path).read().encode(), xml_parser
            )

            self.filtered_tracks = [
                track
                for track in self.xml_content.find("Model")
                .find("FilteredTracks")
                .findall("TrackID")
            ]
            self.filtered_track_ids = [
                int(track.get(self.trackid_key))
                for track in self.xml_content.find("Model")
                .find("FilteredTracks")
                .findall("TrackID")
            ]
            self.max_track_id = max(self.filtered_track_ids)

            self._get_xml_data()
        if not isinstance(self.master_xml_path, str):
            if self.master_xml_path.is_file():

                self.xml_content = et.fromstring(
                    open(self.master_xml_path).read().encode(), xml_parser
                )
                self.filtered_tracks = [
                    track
                    for track in self.xml_content.find("Model")
                    .find("FilteredTracks")
                    .findall("TrackID")
                ]
                self.filtered_track_ids = [
                    int(track.get(self.trackid_key))
                    for track in self.xml_content.find("Model")
                    .find("FilteredTracks")
                    .findall("TrackID")
                ]
                self.max_track_id = max(self.filtered_track_ids)

                self._get_master_xml_data()

    def _create_channel_tree(self):
        self._timed_channel_seg_image = {}
        self.count = 0
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            for i in range(self.channel_seg_image.shape[0]):
                futures.append(executor.submit(self._channel_computer, i))

            if self.progress_bar is not None:

                self.progress_bar.label = "Doing channel computation"
                self.progress_bar.range = (
                    0,
                    len(futures),
                )
                self.progress_bar.show()

            for r in concurrent.futures.as_completed(futures):
                self.count = self.count + 1
                if self.progress_bar is not None:
                    self.progress_bar.value = self.count
                r.result()

    def _channel_computer(self, i):

        if self.image is not None:
            intensity_image = self.image
        else:
            intensity_image = self.channel_seg_image

        properties = measure.regionprops(
            self.channel_seg_image[i, :], intensity_image[i, :]
        )
        centroids = [prop.centroid for prop in properties]
        labels = [prop.label for prop in properties]
        volume = [prop.area for prop in properties]
        intensity_mean = [prop.intensity_mean for prop in properties]
        intensity_total = [prop.intensity_mean * prop.area for prop in properties]
        bounding_boxes = [prop.bbox for prop in properties]

        tree = spatial.cKDTree(centroids)

        self._timed_channel_seg_image[str(i)] = (
            tree,
            centroids,
            labels,
            volume,
            intensity_mean,
            intensity_total,
            bounding_boxes,
        )

    def _get_attributes(self):
        self.Attributeids, self.AllValues = get_spot_dataset(
            self.spot_dataset,
            self.track_analysis_spot_keys,
            self.xcalibration,
            self.ycalibration,
            self.zcalibration,
            self.AttributeBoxname,
            self.detectorchannel,
        )
        self.TrackAttributeids, self.AllTrackValues = get_track_dataset(
            self.track_dataset,
            self.track_analysis_spot_keys,
            self.track_analysis_track_keys,
            self.TrackAttributeBoxname,
        )
        self.AllEdgesValues = get_edges_dataset(
            self.edges_dataset,
            self.edges_dataset_index,
            self.track_analysis_spot_keys,
            self.track_analysis_edges_keys,
        )

    def _get_cell_sizes(self):

        if self.seg_image is not None:
            self.timed_cell_size = get_largest_size(compute_cell_size(self.seg_image))
        elif self.channel_seg_image is not None:
            self.timed_cell_size = get_largest_size(
                compute_cell_size(self.channel_seg_image)
            )
        else:
            self.timed_cell_size = 50
        self.cell_veto_box = 4 * self.timed_cell_size

    def _get_boundary_points(self):

        if self.mask is not None:

            if self.channel_seg_image is not None:

                self.update_mask = check_and_update_mask(
                    self.mask, self.channel_seg_image
                )

            if self.seg_image is not None:

                self.update_mask = check_and_update_mask(self.mask, self.seg_image)

            if self.seg_image is None and self.image is not None:

                self.update_mask = check_and_update_mask(self.mask, self.image)

            self.mask = self.update_mask
            self.timed_mask, self.boundary = boundary_points(
                self.mask, self.xcalibration, self.ycalibration, self.zcalibration
            )
        elif self.mask is None:
            if self.seg_image is not None:

                self.update_mask = np.zeros(self.seg_image.shape, dtype=np.uint8)

            if self.seg_image is None and self.image is not None:

                self.update_mask = np.zeros(self.image.shape, dtype=np.uint8)
            else:
                self.update_mask = np.zeros(self.imagesize, dtype=np.uint8)

            self.mask = self.update_mask
            self.mask[:, :, 1:-1, 1:-1] = 1
            self.timed_mask, self.boundary = boundary_points(
                self.mask, self.xcalibration, self.ycalibration, self.zcalibration
            )

    def _get_track_features(self, track):

        track_displacement = float(track.get(self.displacement_key))
        total_track_distance = float(track.get(self.total_track_distance_key))
        max_track_distance = float(track.get(self.max_distance_traveled_key))
        track_duration = float(track.get(self.track_duration_key))

        return (
            track_displacement,
            total_track_distance,
            max_track_distance,
            track_duration,
        )

    def _generate_generations(self, track):

        all_source_ids = []
        all_target_ids = []

        for edge in track.findall("Edge"):

            source_id = int(edge.get(self.spot_source_id_key))
            target_id = int(edge.get(self.spot_target_id_key))
            all_source_ids.append(source_id)
            all_target_ids.append(target_id)
            if source_id in self.edge_target_lookup.keys():
                self.edge_target_lookup[source_id].append(target_id)
            else:
                self.edge_target_lookup[source_id] = [target_id]
            self.edge_source_lookup[target_id] = source_id

        return all_source_ids, all_target_ids

    def _create_root_leaf(self, all_source_ids: list):

        root_leaf = []
        root_root = []
        for source_id in all_source_ids:
            if source_id in self.edge_source_lookup:
                source_target_id = self.edge_source_lookup[source_id]
            else:
                source_target_id = None
            if source_target_id is None:
                if source_id not in root_root:
                    root_root.append(source_id)

        return root_root, root_leaf

    def _create_generations(self, all_source_ids: list):

        root_leaf = []
        root_root = []
        root_splits = []
        root_pre_leaf = []
        # Get the root id
        for source_id in all_source_ids:
            if source_id in self.edge_source_lookup:
                source_target_id = self.edge_source_lookup[source_id]
            else:
                source_target_id = None
            target_target_id = self.edge_target_lookup[source_id]
            if source_target_id is None:
                if source_id not in root_root:
                    root_root.append(source_id)
            if len(target_target_id) > 1:
                if source_id not in root_splits:
                    root_splits.append(source_id)
            if target_target_id[0] not in self.edge_target_lookup:
                root_leaf.append(target_target_id[0])
                root_pre_leaf.append(source_id)

        if len(list(self.oneat_dividing_tracks.keys())) > 1:
            for cell_id in list(self.oneat_dividing_tracks.keys()):
                if (
                    cell_id in all_source_ids
                    and cell_id not in root_splits
                    and cell_id not in root_leaf
                ):
                    root_splits.append(cell_id)

        return root_root, root_splits, root_leaf

    def _sort_dividing_cells(self, root_splits):
        cell_id_times = []
        split_cell_ids = []
        for root_split in root_splits:
            split_cell_id_time = self.unique_spot_properties[root_split][
                self.frameid_key
            ]
            self.cell_id_times.append(split_cell_id_time)
            self.split_cell_ids.append(root_split)

            cell_id_times.append(split_cell_id_time)
            split_cell_ids.append(root_split)

        sorted_indices = sorted(
            range(len(cell_id_times)), key=lambda k: cell_id_times[k]
        )
        sorted_cell_ids = [split_cell_ids[i] for i in sorted_indices]

        return sorted_cell_ids

    def _iterate_dividing_recursive(
        self,
        root_leaf,
        target_cell,
        sorted_root_splits,
        gen_count,
        tracklet_count,
        tracklet_count_taken,
    ):

        self.generation_dict[target_cell] = gen_count
        self.tracklet_dict[target_cell] = tracklet_count

        if target_cell == root_leaf:
            return

        next_target_cell = None

        if target_cell in sorted_root_splits:
            next_gen_count = gen_count + 1
            if target_cell in self.edge_target_lookup:
                target_cells = self.edge_target_lookup[target_cell]
                for k in range(len(target_cells)):
                    daughter_target_cell = target_cells[k]
                    tracklet_count = tracklet_count + 1 + k
                    tracklet_count = self._unique_tracklet_count(
                        tracklet_count_taken, tracklet_count
                    )
                    tracklet_count_taken.append(tracklet_count)
                    self._iterate_dividing_recursive(
                        root_leaf,
                        daughter_target_cell,
                        sorted_root_splits,
                        next_gen_count,
                        tracklet_count,
                        tracklet_count_taken,
                    )

        if target_cell in self.edge_target_lookup:
            next_target_cells = self.edge_target_lookup[target_cell]
            next_target_cell = next_target_cells[0]

            while next_target_cell not in sorted_root_splits:
                self.generation_dict[next_target_cell] = gen_count
                self.tracklet_dict[next_target_cell] = tracklet_count
                if next_target_cell in root_leaf:
                    self.generation_dict[target_cell] = gen_count
                    self.tracklet_dict[target_cell] = tracklet_count
                    break
                if next_target_cell in self.edge_target_lookup:
                    next_target_cells = self.edge_target_lookup[next_target_cell]
                    next_target_cell = next_target_cells[0]
                    if next_target_cell in root_leaf:
                        self.generation_dict[target_cell] = gen_count
                        self.tracklet_dict[target_cell] = tracklet_count
                        break

        if next_target_cell is not None:
            self.generation_dict[next_target_cell] = gen_count
            self.tracklet_dict[next_target_cell] = tracklet_count
            next_gen_count = gen_count + 1

            if next_target_cell in self.edge_target_lookup:
                target_cells = self.edge_target_lookup[next_target_cell]
                for k in range(len(target_cells)):
                    target_cell = target_cells[k]
                    tracklet_count = tracklet_count + 1 + k
                    tracklet_count = self._unique_tracklet_count(
                        tracklet_count_taken, tracklet_count
                    )
                    tracklet_count_taken.append(tracklet_count)
                    self._iterate_dividing_recursive(
                        root_leaf,
                        target_cell,
                        sorted_root_splits,
                        next_gen_count,
                        tracklet_count,
                        tracklet_count_taken,
                    )

    def _iterate_dividing(self, root_root, root_leaf, root_splits):

        gen_count = 0
        tracklet_count = 0
        tracklet_count_taken = []
        for root_all in root_root:
            self.generation_dict[root_all] = gen_count
            self.tracklet_dict[root_all] = tracklet_count
            tracklet_count_taken.append(tracklet_count)
            if root_all in self.edge_target_lookup and root_all not in root_splits:
                target_cell = self.edge_target_lookup[root_all][0]
                while target_cell not in root_splits:
                    if target_cell in self.edge_target_lookup:
                        self.generation_dict[target_cell] = gen_count
                        self.tracklet_dict[target_cell] = tracklet_count
                        tracklet_count_taken.append(tracklet_count)
                        target_cell = self.edge_target_lookup[target_cell][0]
                    else:
                        self.generation_dict[target_cell] = gen_count
                        self.tracklet_dict[target_cell] = tracklet_count
                        tracklet_count_taken.append(tracklet_count)
                        break
            # Start of track is a dividing cell
            if root_all in self.edge_target_lookup and root_all in root_splits:
                target_cells = self.edge_target_lookup[root_all]
                gen_count = gen_count + 1
                for j in range(len(target_cells)):
                    target_cell = target_cells[j]
                    tracklet_count = tracklet_count + 1 + j
                    tracklet_count = self._unique_tracklet_count(
                        tracklet_count_taken, tracklet_count
                    )
                    tracklet_count_taken.append(tracklet_count)
                    self._iterate_dividing_recursive(
                        root_leaf,
                        target_cell,
                        root_splits,
                        gen_count,
                        tracklet_count,
                        tracklet_count_taken,
                    )

        if len(root_splits) > 0:
            sorted_root_splits = self._sort_dividing_cells(root_splits)
            first_split = sorted_root_splits[0]
            self.generation_dict[first_split] = gen_count
            self.tracklet_dict[first_split] = tracklet_count
            if first_split in self.edge_target_lookup:
                target_cells = self.edge_target_lookup[first_split]
                gen_count = gen_count + 1
                for i in range(len(target_cells)):
                    tracklet_count = tracklet_count + 1 + i
                    tracklet_count = self._unique_tracklet_count(
                        tracklet_count_taken, tracklet_count
                    )
                    tracklet_count_taken.append(tracklet_count)
                    target_cell = target_cells[i]
                    self._iterate_dividing_recursive(
                        root_leaf,
                        target_cell,
                        sorted_root_splits,
                        gen_count,
                        tracklet_count,
                        tracklet_count_taken,
                    )

    def _unique_tracklet_count(self, tracklet_count_taken, tracklet_count):

        while tracklet_count in tracklet_count_taken:
            tracklet_count += 1
            if tracklet_count not in tracklet_count_taken:
                break
        return tracklet_count

    def _iterate_split_down(self, root_root, root_leaf, root_splits):

        self._iterate_dividing(root_root, root_leaf, root_splits)

    
    def _get_label_density(self, frame, test_location):
        """
        Compute the label density in a local neighborhood for both 2D+t and 3D+t images.
        
        Args:
            frame (int): The time frame index.
            test_location (tuple): The (z, y, x) coordinates for 3D+t or (y, x) for 2D+t.

        Returns:
            int: The local cell density (number of unique labels in the region).
        """
        
        # Select the correct frame
        if self.channel_seg_image is None:
            current_frame_image = self.seg_image[int(float(frame)), :]
        else:
            current_frame_image = self.channel_seg_image[int(float(frame)), :]

        # Check if the image is 2D+t or 3D+t
        is_3d = current_frame_image.ndim == 3  # True if (z, y, x), False if (y, x)

        if is_3d:
            # 3D+t case
            z_test, y_test, x_test = test_location
            min_z = max(0, int(z_test - self.cell_veto_box))
            max_z = min(current_frame_image.shape[0] - 1, int(z_test + self.cell_veto_box))
            min_y = max(0, int(y_test - self.cell_veto_box))
            max_y = min(current_frame_image.shape[1] - 1, int(y_test + self.cell_veto_box))
            min_x = max(0, int(x_test - self.cell_veto_box))
            max_x = min(current_frame_image.shape[2] - 1, int(x_test + self.cell_veto_box))
            
            subvolume = current_frame_image[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

        else:
            # 2D+t case
            z_test, y_test, x_test = test_location
            min_y = max(0, int(y_test - self.cell_veto_box))
            max_y = min(current_frame_image.shape[0] - 1, int(y_test + self.cell_veto_box))
            min_x = max(0, int(x_test - self.cell_veto_box))
            max_x = min(current_frame_image.shape[1] - 1, int(x_test + self.cell_veto_box))

            subvolume = current_frame_image[min_y:max_y+1, min_x:max_x+1]

        # Compute unique labels
        unique_labels = np.unique(subvolume)
        local_cell_density = len(unique_labels)

        return local_cell_density

    def _get_boundary_dist(self, frame, testlocation):

        if self.mask is not None:

            tree, indices, maskcentroid = self.timed_mask[str(int(float(frame)))]

            distance_cell_mask, locationindex = tree.query(testlocation)
            distance_cell_mask = max(0, distance_cell_mask)

        else:
            distance_cell_mask = 0
            maskcentroid = (1, 1, 1)

        return distance_cell_mask, maskcentroid

    def _track_computer(self, track, track_id):

        current_cell_ids = []

        (
            track_displacement,
            total_track_distance,
            max_track_distance,
            track_duration,
        ) = self._get_track_features(track)

        all_source_ids, all_target_ids = self._generate_generations(track)
        root_root, root_splits, root_leaf = self._create_generations(all_source_ids)
        self._iterate_split_down(root_root, root_leaf, root_splits)

        number_dividing = len(root_splits)
        # Determine if a track has divisions or none
        if len(root_splits) > 0:
            self.unique_track_mitosis_label[track_id] = [1, number_dividing]
            dividing_trajectory = True
            if int(track_id) not in self.AllTrackIds:
                self.AllTrackIds.append(int(track_id))
            if int(track_id) not in self.DividingTrackIds:
                self.DividingTrackIds.append(int(track_id))

        else:
            self.unique_track_mitosis_label[track_id] = [0, 0]
            dividing_trajectory = False
            if int(track_id) not in self.AllTrackIds:
                self.AllTrackIds.append(int(track_id))
            if int(track_id) not in self.NormalTrackIds:
                self.NormalTrackIds.append(int(track_id))

        for leaf in root_leaf:
            source_leaf = self.edge_source_lookup[leaf]
            current_cell_ids.append(leaf)
            self._dict_update(leaf, track_id, source_leaf, None)
            self._msd_update(root_root[0], leaf)

            self.unique_spot_properties[leaf].update(
                {self.dividing_key: dividing_trajectory}
            )
            self.unique_spot_properties[leaf].update({self.fate_key: -1})
            self.unique_spot_properties[leaf].update(
                {self.number_dividing_key: number_dividing}
            )
            self.unique_spot_properties[leaf].update(
                {self.displacement_key: track_displacement}
            )
            self.unique_spot_properties[leaf].update(
                {self.total_track_distance_key: total_track_distance}
            )
            self.unique_spot_properties[leaf].update(
                {self.max_distance_traveled_key: max_track_distance}
            )
            self.unique_spot_properties[leaf].update(
                {self.track_duration_key: track_duration}
            )

        for source_id in all_source_ids:
            target_ids = self.edge_target_lookup[source_id]
            current_cell_ids.append(source_id)
            self._msd_update(root_root[0], source_id)
            # Root
            self.unique_spot_properties[source_id].update(
                {self.dividing_key: dividing_trajectory}
            )
            self.unique_spot_properties[source_id].update({self.fate_key: -1})

            self.unique_spot_properties[source_id].update(
                {self.number_dividing_key: number_dividing}
            )
            self.unique_spot_properties[source_id].update(
                {self.displacement_key: track_displacement}
            )
            self.unique_spot_properties[source_id].update(
                {self.total_track_distance_key: total_track_distance}
            )
            self.unique_spot_properties[source_id].update(
                {self.max_distance_traveled_key: max_track_distance}
            )
            self.unique_spot_properties[source_id].update(
                {self.track_duration_key: track_duration}
            )
            if source_id not in all_target_ids:

                for target_id in target_ids:
                    self._dict_update(source_id, track_id, None, target_id)
                    self.unique_spot_properties[target_id].update(
                        {self.dividing_key: dividing_trajectory}
                    )
                    self.unique_spot_properties[target_id].update({self.fate_key: -1})
                    self.unique_spot_properties[target_id].update(
                        {self.number_dividing_key: number_dividing}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.displacement_key: track_displacement}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.total_track_distance_key: total_track_distance}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.max_distance_traveled_key: max_track_distance}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.track_duration_key: track_duration}
                    )
            else:
                # Normal
                source_source_id = self.edge_source_lookup[source_id]
                for target_id in target_ids:
                    self._dict_update(
                        source_id,
                        track_id,
                        source_source_id,
                        target_id,
                    )
                    self.unique_spot_properties[target_id].update(
                        {self.dividing_key: dividing_trajectory}
                    )
                    self.unique_spot_properties[target_id].update({self.fate_key: -1})

                    self.unique_spot_properties[target_id].update(
                        {self.number_dividing_key: number_dividing}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.displacement_key: track_displacement}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.total_track_distance_key: total_track_distance}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.max_distance_traveled_key: max_track_distance}
                    )
                    self.unique_spot_properties[source_id].update(
                        {self.track_duration_key: track_duration}
                    )

        for current_root in root_root:
            self.root_spots[int(current_root)] = self.unique_spot_properties[
                int(current_root)
            ]
            self._msd_update(root_root[0], current_root)

        self.all_current_cell_ids[int(track_id)] = current_cell_ids
        for i in range(len(current_cell_ids)):

            k = int(current_cell_ids[i])
            all_dict_values = self.unique_spot_properties[k]

            t = int(float(all_dict_values[self.frameid_key]))
            z = float(all_dict_values[self.zposid_key])
            y = float(all_dict_values[self.yposid_key])
            x = float(all_dict_values[self.xposid_key])

            spot_centroid = (
                round(z) / self.zcalibration,
                round(y) / self.ycalibration,
                round(x) / self.xcalibration,
            )
            frame_spot_centroid = (
                t,
                round(z) / self.zcalibration,
                round(y) / self.ycalibration,
                round(x) / self.xcalibration,
            )

            self.unique_track_centroid[frame_spot_centroid] = track_id
            self.unique_spot_centroid[frame_spot_centroid] = k

            if str(t) in self._timed_centroid:
                tree, spot_centroids = self._timed_centroid[str(t)]
                spot_centroids.append(spot_centroid)
                tree = spatial.cKDTree(spot_centroids)
                self._timed_centroid[str(t)] = tree, spot_centroids
            else:
                spot_centroids = []
                spot_centroids.append(spot_centroid)
                tree = spatial.cKDTree(spot_centroids)
                self._timed_centroid[str(t)] = tree, spot_centroids

    def _master_track_computer(self, track, track_id, t_start=None, t_end=None):
        current_cell_ids = []
        (
            track_displacement,
            total_track_distance,
            max_track_distance,
            track_duration,
        ) = self._get_track_features(track)

        all_source_ids, all_target_ids = self._generate_generations(track)
        root_root, root_splits, root_leaf = self._create_generations(all_source_ids)
        self._iterate_split_down(root_root, root_leaf, root_splits)
        # Determine if a track has divisions or none
        number_dividing = len(root_splits)
        if number_dividing > 0:
            self.unique_track_mitosis_label[track_id] = [1, number_dividing]
            dividing_trajectory = True
            if int(track_id) not in self.AllTrackIds:
                self.AllTrackIds.append(int(track_id))
            if int(track_id) not in self.DividingTrackIds:
                self.DividingTrackIds.append(int(track_id))

        else:
            self.unique_track_mitosis_label[track_id] = [0, 0]
            dividing_trajectory = False
            if int(track_id) not in self.AllTrackIds:
                self.AllTrackIds.append(int(track_id))
            if int(track_id) not in self.NormalTrackIds:
                self.NormalTrackIds.append(int(track_id))

        for leaf in root_leaf:
            self._second_channel_update(leaf, track_id)
            current_cell_ids.append(leaf)
            self._msd_update(root_root[0], leaf)
            self.unique_spot_properties[leaf].update(
                {self.dividing_key: dividing_trajectory}
            )
            self.unique_spot_properties[leaf].update({self.fate_key: -1})
            self.unique_spot_properties[leaf].update(
                {self.number_dividing_key: number_dividing}
            )
            self.unique_spot_properties[leaf].update(
                {self.displacement_key: track_displacement}
            )
            self.unique_spot_properties[leaf].update(
                {self.total_track_distance_key: total_track_distance}
            )
            self.unique_spot_properties[leaf].update(
                {self.max_distance_traveled_key: max_track_distance}
            )
            self.unique_spot_properties[leaf].update(
                {self.track_duration_key: track_duration}
            )

        for source_id in all_source_ids:
            self._second_channel_update(source_id, track_id)
            self.unique_spot_properties[source_id].update(
                {self.dividing_key: dividing_trajectory}
            )
            self.unique_spot_properties[source_id].update({self.fate_key: -1})

            self.unique_spot_properties[source_id].update(
                {self.number_dividing_key: number_dividing}
            )
            self.unique_spot_properties[source_id].update(
                {self.displacement_key: track_displacement}
            )
            self.unique_spot_properties[source_id].update(
                {self.total_track_distance_key: total_track_distance}
            )
            self.unique_spot_properties[source_id].update(
                {self.max_distance_traveled_key: max_track_distance}
            )
            self.unique_spot_properties[source_id].update(
                {self.track_duration_key: track_duration}
            )
            current_cell_ids.append(source_id)
            self._msd_update(root_root[0], source_id)

        for current_root in root_root:
            self._second_channel_update(current_root, track_id)
            self.root_spots[int(current_root)] = self.unique_spot_properties[
                int(current_root)
            ]
            self.unique_spot_properties[source_id].update(
                {self.dividing_key: dividing_trajectory}
            )
            self.unique_spot_properties[source_id].update({self.fate_key: -1})

            self.unique_spot_properties[source_id].update(
                {self.number_dividing_key: number_dividing}
            )
            self.unique_spot_properties[source_id].update(
                {self.displacement_key: track_displacement}
            )
            self.unique_spot_properties[source_id].update(
                {self.total_track_distance_key: total_track_distance}
            )
            self.unique_spot_properties[source_id].update(
                {self.max_distance_traveled_key: max_track_distance}
            )
            self.unique_spot_properties[source_id].update(
                {self.track_duration_key: track_duration}
            )
            self._msd_update(root_root[0], current_root)

        self.all_current_cell_ids[int(track_id)] = current_cell_ids

        for i in range(len(current_cell_ids)):

            k = int(current_cell_ids[i])

            all_dict_values = self.unique_spot_properties[k]

            t = int(float(all_dict_values[self.frameid_key]))
            z = float(all_dict_values[self.zposid_key])
            y = float(all_dict_values[self.yposid_key])
            x = float(all_dict_values[self.xposid_key])

            spot_centroid = (
                round(z) / self.zcalibration,
                round(y) / self.ycalibration,
                round(x) / self.xcalibration,
            )
            frame_spot_centroid = (
                t,
                round(z) / self.zcalibration,
                round(y) / self.ycalibration,
                round(x) / self.xcalibration,
            )

            self.unique_spot_centroid[frame_spot_centroid] = k
            self.unique_track_centroid[frame_spot_centroid] = track_id

            if t_start is not None and t_end is not None:
                if t >= t_start and t <= t_end:
                    if str(t) in self._timed_centroid:
                        tree, spot_centroids = self._timed_centroid[str(t)]
                        spot_centroids.append(spot_centroid)
                        tree = spatial.cKDTree(spot_centroids)
                        self._timed_centroid[str(t)] = tree, spot_centroids
                    else:
                        spot_centroids = []
                        spot_centroids.append(spot_centroid)
                        tree = spatial.cKDTree(spot_centroids)
                        self._timed_centroid[str(t)] = tree, spot_centroids
            else:

                if str(t) in self._timed_centroid:
                    tree, spot_centroids = self._timed_centroid[str(t)]
                    spot_centroids.append(spot_centroid)
                    tree = spatial.cKDTree(spot_centroids)
                    self._timed_centroid[str(t)] = tree, spot_centroids
                else:
                    spot_centroids = []
                    spot_centroids.append(spot_centroid)
                    tree = spatial.cKDTree(spot_centroids)
                    self._timed_centroid[str(t)] = tree, spot_centroids

    def _second_channel_update(self, cell_id, track_id):

        if self.channel_seg_image is not None:

            frame = self.unique_spot_properties[int(cell_id)][self.frameid_key]
            z = (
                self.unique_spot_properties[int(cell_id)][self.zposid_key]
                / self.zcalibration
            )
            y = (
                self.unique_spot_properties[int(cell_id)][self.yposid_key]
                / self.ycalibration
            )
            x = (
                self.unique_spot_properties[int(cell_id)][self.xposid_key]
                / self.xcalibration
            )
            self._second_channel_spots(frame, z, y, x, cell_id, track_id)

    def get_track_dataframe(self):
        """
        Computes a DataFrame for all track_ids containing T, Z, Y, X, and all properties
        using self.unique_tracks and self.unique_track_properties.

        Returns:
            pd.DataFrame: DataFrame containing all track properties for all track_ids.
        """
        data_list = []  

        column_names = [
            "unique_id", "t", "z", "y", "x", 
            "gen_id", "radius", "eccentricity_comp_first", "eccentricity_comp_second", 
            "eccentricity_comp_third", "surface_area", "total_intensity", "speed", 
            "motion_angle_z", "motion_angle_y", "motion_angle_x", "acceleration", 
            "distance_cell_mask", "local_cell_density", "radial_angle_z", "radial_angle_y", 
            "radial_angle_x", "cell_axis_z", "cell_axis_y", "cell_axis_x", "track_displacement", 
            "total_track_distance", "max_track_distance", "track_duration", "msd"
        ]
        
        for track_id in self.unique_tracks:
            current_tracklets = self.unique_tracks[track_id]
            current_tracklets_properties = self.unique_track_properties[track_id]
            
            for idx in range(len(current_tracklets)):
                tracklet_info = current_tracklets[idx]
                properties_info = current_tracklets_properties[idx]
                
                properties_info_skipped = properties_info[2:]
                
                combined_info = list(tracklet_info) + list(properties_info_skipped)
                
                data_list.append(combined_info)

        track_df = pd.DataFrame(data_list, columns=column_names)

        return track_df





    def _final_tracks(self, track_id):

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

        current_tracklets = np.asarray(
            current_tracklets[str(track_id)], dtype=np.float32
        )
        current_tracklets_properties = np.asarray(
            current_tracklets_properties[str(track_id)], dtype=np.float32
        )

        self.unique_tracks[track_id] = current_tracklets
        self.unique_track_properties[track_id] = current_tracklets_properties

    def _tracklet_and_properties(
        self,
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
    ):

        gen_id = int(float(all_dict_values[self.generationid_key]))
        speed = float(all_dict_values[self.speed_key])
        msd = 0
        if self.msd_key in all_dict_values.keys():
            msd = float(all_dict_values[self.msd_key])
        acceleration = float(all_dict_values[self.acceleration_key])
        motion_angle_z = float(all_dict_values[self.motion_angle_z_key])
        motion_angle_y = float(all_dict_values[self.motion_angle_y_key])
        motion_angle_x = float(all_dict_values[self.motion_angle_x_key])

        radial_angle_z = float(all_dict_values[self.radial_angle_z_key])
        radial_angle_y = float(all_dict_values[self.radial_angle_y_key])
        radial_angle_x = float(all_dict_values[self.radial_angle_x_key])

        radius = float(all_dict_values[self.radius_key])
        total_intensity = float(all_dict_values[self.total_intensity_key])

        distance_cell_mask = float(all_dict_values[self.distance_cell_mask_key])
        local_cell_density = float(all_dict_values[self.local_cell_density_key])

        track_displacement = float(all_dict_values[self.displacement_key])
        total_track_distance = float(all_dict_values[self.total_track_distance_key])
        max_track_distance = float(all_dict_values[self.max_distance_traveled_key])
        track_duration = float(all_dict_values[self.track_duration_key])

        if self.latent_shape_features_key in all_dict_values.keys():
            latent_shape_features = list(
                all_dict_values[self.latent_shape_features_key]
            )
            compute_with_latent_features = True
        else:
            latent_shape_features = [-1] * self.latent_features
            compute_with_latent_features = False

        if self.surface_area_key in all_dict_values.keys():

            eccentricity_comp_first = float(
                all_dict_values[self.eccentricity_comp_firstkey]
            )
            eccentricity_comp_second = float(
                all_dict_values[self.eccentricity_comp_secondkey]
            )
            eccentricity_comp_third = float(
                all_dict_values[self.eccentricity_comp_thirdkey]
            )
            surface_area = float(all_dict_values[self.surface_area_key])
            cell_axis_z = float(all_dict_values[self.cell_axis_z_key])
            cell_axis_y = float(all_dict_values[self.cell_axis_y_key])
            cell_axis_x = float(all_dict_values[self.cell_axis_x_key])

        else:
            eccentricity_comp_first = -1
            eccentricity_comp_second = -1
            eccentricity_comp_third = -1
            surface_area = -1
            cell_axis_z = -1
            cell_axis_y = -1
            cell_axis_x = -1

        frame_spot_centroid = (
            t,
            round(z) / self.zcalibration,
            round(y) / self.ycalibration,
            round(x) / self.xcalibration,
        )
        self.unique_spot_centroid[frame_spot_centroid] = k

        if current_track_id in current_tracklets:
            tracklet_array = current_tracklets[current_track_id]
            current_tracklet_array = np.array(
                [
                    int(float(unique_id)),
                    t,
                    z / self.zcalibration,
                    y / self.ycalibration,
                    x / self.xcalibration,
                ]
            )
            current_tracklets[current_track_id] = np.vstack(
                (tracklet_array, current_tracklet_array)
            )

            value_array = current_tracklets_properties[current_track_id]
            current_value_list = [
                t,
                int(float(unique_id)),
                gen_id,
                radius,
                eccentricity_comp_first,
                eccentricity_comp_second,
                eccentricity_comp_third,
                surface_area,
                total_intensity,
                speed,
                motion_angle_z,
                motion_angle_y,
                motion_angle_x,
                acceleration,
                distance_cell_mask,
                local_cell_density,
                radial_angle_z,
                radial_angle_y,
                radial_angle_x,
                cell_axis_z,
                cell_axis_y,
                cell_axis_x,
                track_displacement,
                total_track_distance,
                max_track_distance,
                track_duration,
                msd,
            ]

            if compute_with_latent_features:
                current_value_list.extend(latent_shape_features)

            current_value_array = np.array(current_value_list)
            diff = value_array.shape[-1] - current_value_array.shape[-1]
            if diff > 0:
                current_value_array = np.pad(
                    current_value_array, (0, diff), mode="constant", constant_values=-1
                )
            if diff < 0:
                value_array = np.pad(
                    value_array, (0, -diff), mode="constant", constant_values=-1
                )
            current_tracklets_properties[current_track_id] = np.vstack(
                (value_array, current_value_array)
            )

        else:
            current_tracklet_array = np.array(
                [
                    int(float(unique_id)),
                    t,
                    z / self.zcalibration,
                    y / self.ycalibration,
                    x / self.xcalibration,
                ]
            )
            current_tracklets[current_track_id] = current_tracklet_array

            current_value_list = [
                t,
                int(float(unique_id)),
                gen_id,
                radius,
                eccentricity_comp_first,
                eccentricity_comp_second,
                eccentricity_comp_third,
                surface_area,
                total_intensity,
                speed,
                motion_angle_z,
                motion_angle_y,
                motion_angle_x,
                acceleration,
                distance_cell_mask,
                local_cell_density,
                radial_angle_z,
                radial_angle_y,
                radial_angle_x,
                cell_axis_z,
                cell_axis_y,
                cell_axis_x,
                track_displacement,
                total_track_distance,
                max_track_distance,
                track_duration,
                msd,
            ]
            if compute_with_latent_features:
                current_value_list.extend(latent_shape_features)

            current_value_array = np.array(current_value_list)
            current_tracklets_properties[current_track_id] = current_value_array

        return current_tracklets, current_tracklets_properties

    def _master_spot_computer(self, frame):

        for Spotobject in frame.findall("Spot"):

            cell_id = int(Spotobject.get(self.spotid_key))

            if self.uniqueid_key in Spotobject.keys():

                radius = float(Spotobject.get(self.radius_key))
                quality = float(Spotobject.get(self.quality_key))
                total_intensity = float(Spotobject.get(self.total_intensity_key))
                mean_intensity = float(Spotobject.get(self.mean_intensity_key))

                self.tracklet_id_to_trackmate_id[
                    float(str(Spotobject.get(self.uniqueid_key)))
                ] = int(float(str(Spotobject.get(self.trackid_key))))

                self.tracklet_id_to_generation_id[
                    float(str(Spotobject.get(self.uniqueid_key)))
                ] = int(float(str(Spotobject.get(self.generationid_key))))

                self.tracklet_id_to_tracklet_number_id[
                    float(str(Spotobject.get(self.uniqueid_key)))
                ] = int(float(str(Spotobject.get(self.trackletid_key))))

                self.unique_spot_properties[cell_id] = {
                    self.cellid_key: int(float(Spotobject.get(self.spotid_key))),
                    self.frameid_key: int(float(Spotobject.get(self.frameid_key))),
                    self.zposid_key: float(Spotobject.get(self.zposid_key)),
                    self.yposid_key: float(Spotobject.get(self.yposid_key)),
                    self.xposid_key: float(Spotobject.get(self.xposid_key)),
                    self.total_intensity_key: total_intensity,
                    self.mean_intensity_key: mean_intensity,
                    self.radius_key: radius,
                    self.quality_key: quality,
                    self.distance_cell_mask_key: (
                        float(Spotobject.get(self.distance_cell_mask_key))
                    ),
                    self.local_cell_density_key: (
                        float(Spotobject.get(self.local_cell_density_key))
                    ),
                    self.uniqueid_key: str(Spotobject.get(self.uniqueid_key)),
                    self.trackletid_key: str(Spotobject.get(self.trackletid_key)),
                    self.generationid_key: str(Spotobject.get(self.generationid_key)),
                    self.trackid_key: str(Spotobject.get(self.trackid_key)),
                    self.motion_angle_z_key: (
                        float(Spotobject.get(self.motion_angle_z_key))
                    ),
                    self.motion_angle_y_key: (
                        float(Spotobject.get(self.motion_angle_y_key))
                    ),
                    self.motion_angle_x_key: (
                        float(Spotobject.get(self.motion_angle_x_key))
                    ),
                    self.speed_key: (float(Spotobject.get(self.speed_key))),
                    self.acceleration_key: (
                        float(Spotobject.get(self.acceleration_key))
                    ),
                    self.radial_angle_z_key: float(
                        Spotobject.get(self.radial_angle_z_key)
                    ),
                    self.radial_angle_y_key: float(
                        Spotobject.get(self.radial_angle_y_key)
                    ),
                    self.radial_angle_x_key: float(
                        Spotobject.get(self.radial_angle_x_key)
                    ),
                    self.number_dividing_key: float(
                        Spotobject.get(self.number_dividing_key)
                    ),
                    self.dividing_key: bool(Spotobject.get(self.dividing_key)),
                    self.displacement_key: float(Spotobject.get(self.displacement_key)),
                    self.total_track_distance_key: float(
                        Spotobject.get(self.total_track_distance_key)
                    ),
                    self.max_distance_traveled_key: float(
                        Spotobject.get(self.max_distance_traveled_key)
                    ),
                    self.track_duration_key: float(
                        Spotobject.get(self.track_duration_key)
                    ),
                }
                if self.surface_area_key in Spotobject.keys():
                    self.unique_spot_properties[int(cell_id)].update(
                        {
                            self.eccentricity_comp_firstkey: float(
                                Spotobject.get(self.eccentricity_comp_firstkey)
                            ),
                            self.eccentricity_comp_secondkey: float(
                                Spotobject.get(self.eccentricity_comp_secondkey)
                            ),
                            self.eccentricity_comp_thirdkey: float(
                                Spotobject.get(self.eccentricity_comp_thirdkey)
                            ),
                            self.surface_area_key: float(
                                Spotobject.get(self.surface_area_key)
                            ),
                            self.cell_axis_z_key: float(
                                Spotobject.get(self.cell_axis_z_key)
                            ),
                            self.cell_axis_y_key: float(
                                Spotobject.get(self.cell_axis_y_key)
                            ),
                            self.cell_axis_x_key: float(
                                Spotobject.get(self.cell_axis_x_key)
                            ),
                        }
                    )

            elif self.uniqueid_key not in Spotobject.keys():

                if self.detectorchannel == 1:
                    TOTAL_INTENSITY = Spotobject.get(self.total_intensity_ch2_key)
                    MEAN_INTENSITY = Spotobject.get(self.mean_intensity_ch2_key)
                else:
                    TOTAL_INTENSITY = Spotobject.get(self.total_intensity_ch1_key)
                    MEAN_INTENSITY = Spotobject.get(self.mean_intensity_ch1_key)
                RADIUS = float(Spotobject.get(self.radius_key))
                QUALITY = float(Spotobject.get(self.quality_key))
                TOTAL_INTENSITY = float(TOTAL_INTENSITY)
                MEAN_INTENSITY = float(MEAN_INTENSITY)
                self.unique_spot_properties[cell_id] = {
                    self.cellid_key: int(cell_id),
                    self.frameid_key: int(float(Spotobject.get(self.frameid_key))),
                    self.zposid_key: float(Spotobject.get(self.zposid_key)),
                    self.yposid_key: float(Spotobject.get(self.yposid_key)),
                    self.xposid_key: float(Spotobject.get(self.xposid_key)),
                    self.total_intensity_key: TOTAL_INTENSITY,
                    self.mean_intensity_key: MEAN_INTENSITY,
                    self.radius_key: RADIUS,
                    self.quality_key: QUALITY,
                }

            frame_spot_centroid = (
                int(float(Spotobject.get(self.frameid_key))),
                float(Spotobject.get(self.zposid_key)) / self.zcalibration,
                float(Spotobject.get(self.yposid_key)) / self.ycalibration,
                float(Spotobject.get(self.xposid_key)) / self.xcalibration,
            )
            self.unique_oneat_spot_centroid[frame_spot_centroid] = cell_id

    def _spot_computer(self, frame):

        for Spotobject in frame.findall("Spot"):
            # Create object with unique cell ID
            cell_id = int(Spotobject.get(self.spotid_key))
            # Get the TZYX location of the cells in that frame
            if self.detectorchannel == 1:
                if Spotobject.get(self.total_intensity_ch2_key) is not None:
                    TOTAL_INTENSITY = float(
                        Spotobject.get(self.total_intensity_ch2_key)
                    )
                    MEAN_INTENSITY = float(Spotobject.get(self.mean_intensity_ch2_key))
                else:
                    TOTAL_INTENSITY = -1
                    MEAN_INTENSITY = -1
            else:
                if Spotobject.get(self.total_intensity_ch1_key) is not None:
                    TOTAL_INTENSITY = float(
                        Spotobject.get(self.total_intensity_ch1_key)
                    )
                    MEAN_INTENSITY = float(Spotobject.get(self.mean_intensity_ch1_key))
                else:
                    TOTAL_INTENSITY = -1
                    MEAN_INTENSITY = -1

            RADIUS = float(Spotobject.get(self.radius_key))
            QUALITY = float(Spotobject.get(self.quality_key))
            testlocation = (
                float(Spotobject.get(self.zposid_key)),
                float(Spotobject.get(self.yposid_key)),
                float(Spotobject.get(self.xposid_key)),
            )
            frame = Spotobject.get(self.frameid_key)
            distance_cell_mask, maskcentroid = self._get_boundary_dist(
                frame, testlocation
            )
            if self.seg_image is not None or self.channel_seg_image is not None:
               local_cell_density = self._get_label_density(frame, testlocation)
            else:
                local_cell_density = 1
            self.unique_spot_properties[cell_id] = {
                self.cellid_key: int(cell_id),
                self.frameid_key: int(float(Spotobject.get(self.frameid_key))),
                self.zposid_key: float(Spotobject.get(self.zposid_key)),
                self.yposid_key: float(Spotobject.get(self.yposid_key)),
                self.xposid_key: float(Spotobject.get(self.xposid_key)),
                self.total_intensity_key: TOTAL_INTENSITY,
                self.mean_intensity_key: MEAN_INTENSITY,
                self.radius_key: RADIUS,
                self.quality_key: QUALITY,
                self.distance_cell_mask_key: float(distance_cell_mask),
                self.maskcentroid_z_key: float(maskcentroid[0]),
                self.maskcentroid_y_key: float(maskcentroid[1]),
                self.maskcentroid_x_key: float(maskcentroid[2]),
                self.local_cell_density_key: float(local_cell_density),
            }

            frame_spot_centroid = (
                int(float(Spotobject.get(self.frameid_key))),
                float(Spotobject.get(self.zposid_key)) / self.zcalibration,
                float(Spotobject.get(self.yposid_key)) / self.ycalibration,
                float(Spotobject.get(self.xposid_key)) / self.xcalibration,
            )
            self.unique_oneat_spot_centroid[frame_spot_centroid] = cell_id

    def _get_master_xml_data(self):
        if self.channel_seg_image is not None:
            self.channel_xml_content = self.xml_content
            self.xml_tree = et.parse(self.xml_path)
            self.xml_root = self.xml_tree.getroot()
            base_name = os.path.splitext(os.path.basename(self.xml_path))[0]
            if "nuclei" in base_name:
                base_name = base_name.replace("nuclei", self.second_channel_name)
                new_name = base_name
            else:
                new_name = base_name + f"_{self.second_channel_name}"
            self.channel_xml_name = new_name + ".xml"
            self._create_channel_tree()

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
        self.unique_tracklet_ids = []
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
        self.tcalibration = float(self.settings.get("timeinterval"))
        if self.tcalibration == 0: self.tcalibration = 1
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

        self.count = 0
        futures = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:

            for frame in self.Spotobjects.findall("SpotsInFrame"):
                futures.append(executor.submit(self._master_spot_computer, frame))
            if self.progress_bar is not None:

                self.progress_bar.label = "Collecting Spots"
                self.progress_bar.range = (
                    0,
                    len(futures),
                )
                self.progress_bar.show()

            for r in concurrent.futures.as_completed(futures):
                self.count = self.count + 1
                if self.progress_bar is not None:
                    self.progress_bar.value = self.count
                r.result()

        self._correct_track_status()
        print(f"Iterating over tracks {len(self.filtered_track_ids)}")
        self.count = 0
        futures = []
        if self.progress_bar is not None:
            self.progress_bar.label = "Collecting Tracks"
            self.progress_bar.range = (0, len(self.filtered_track_ids))
            self.progress_bar.show()

        for track in tqdm(self.tracks.findall("Track")):
            track_id = int(track.get(self.trackid_key))
            if track_id in self.filtered_track_ids:
                self._master_track_computer(track, track_id)
                self.count += 1
                if self.progress_bar is not None:
                    self.progress_bar.value = self.count
        if self.channel_seg_image is not None:

            self._create_second_channel_xml()

        for (k, v) in self.graph_split.items():

            daughter_track_id = int(
                float(
                    str(self.unique_spot_properties[int(float(k))][self.uniqueid_key])
                )
            )
            parent_track_id = int(
                float(
                    str(self.unique_spot_properties[int(float(v))][self.uniqueid_key])
                )
            )
            self.graph_tracks[daughter_track_id] = parent_track_id
        self._get_cell_fate_tracks()
        if (
            self.spot_csv_path is not None
            and self.track_csv_path is not None
            and self.edges_csv_path is not None
        ):
            self._get_attributes()
        if self.autoencoder_model is not None:
            self._compute_latent_space()
        self.count = 0
        for index, track_id in enumerate(self.filtered_track_ids):
            if self.progress_bar is not None:
                self.progress_bar.label = "Just one more thing"
                self.progress_bar.range = (
                    0,
                    len(self.filtered_track_ids),
                )
                self.progress_bar.show()
                self.count = self.count + 1
                self.progress_bar.value = self.count
            track = self.filtered_tracks[index]
            if self.channel_seg_image is None:
                self._final_tracks(track_id)
        if self.channel_seg_image is None:
            self._compute_phenotypes()
            self._temporal_plots_trackmate()

    def _create_second_channel_xml(self):

        channel_filtered_tracks = []
        file_name = self.settings.get("filename")
        if "nuclei" in file_name:
            file_name = file_name.replace("nuclei", self.second_channel_name)
        for Spotobject in self.xml_root.iter("ImageData"):
            Spotobject.set("filename", file_name)
        for Spotobject in self.xml_root.iter("Spot"):
            cell_id = int(Spotobject.get(self.spotid_key))
            if cell_id in self.channel_unique_spot_properties.keys():

                new_positionx = self.channel_unique_spot_properties[cell_id][
                    self.xposid_key
                ]
                new_positiony = self.channel_unique_spot_properties[cell_id][
                    self.yposid_key
                ]
                new_positionz = self.channel_unique_spot_properties[cell_id][
                    self.zposid_key
                ]

                new_total_intensity = self.channel_unique_spot_properties[cell_id][
                    self.total_intensity_key
                ]
                new_mean_intensity = self.channel_unique_spot_properties[cell_id][
                    self.mean_intensity_key
                ]

                new_radius = self.channel_unique_spot_properties[cell_id][
                    self.radius_key
                ]
                new_quality = self.channel_unique_spot_properties[cell_id][
                    self.quality_key
                ]
                new_distance_cell_mask = self.channel_unique_spot_properties[cell_id][
                    self.distance_cell_mask_key
                ]

                new_local_density = self.channel_unique_spot_properties[cell_id][
                    self.local_cell_density_key
                ]

                Spotobject.set(self.xposid_key, str(new_positionx))
                Spotobject.set(self.yposid_key, str(new_positiony))
                Spotobject.set(self.zposid_key, str(new_positionz))

                Spotobject.set(self.total_intensity_key, str(new_total_intensity))
                Spotobject.set(self.mean_intensity_key, str(new_mean_intensity))
                Spotobject.set(self.radius_key, str(new_radius))
                Spotobject.set(self.quality_key, str(new_quality))
                Spotobject.set(self.distance_cell_mask_key, str(new_distance_cell_mask))
                Spotobject.set(self.local_cell_density_key, str(new_local_density))
                track_id = self.channel_unique_spot_properties[int(cell_id)][
                    self.trackid_key
                ]
                channel_filtered_tracks.append(track_id)
        print(
            f"Writing new xml at path {self.channel_xml_path}, {self.channel_xml_name}"
        )
        self.xml_tree.write(os.path.join(self.channel_xml_path, self.channel_xml_name))

    def _get_xml_data(self):

        if self.channel_seg_image is not None:
            print(
                f"Segmentation image in second channel {self.second_channel_name} of shape {self.channel_seg_image.shape}"
            )
            self.channel_xml_content = self.xml_content
            self.xml_tree = et.parse(self.xml_path)
            self.xml_root = self.xml_tree.getroot()
            base_name = os.path.splitext(os.path.basename(self.xml_path))[0]

            if "nuclei" in base_name:
                base_name = base_name.replace("nuclei", self.second_channel_name)
                new_name = base_name
            else:
                new_name = base_name + f"_{self.second_channel_name}"
            self.channel_xml_name = new_name + ".xml"
            self._create_channel_tree()
        if (
            self.autoencoder_model is not None or self.enhance_trackmate_xml
        ) and self.seg_image is not None:
            self.master_xml_content = self.xml_content
            self.master_xml_tree = et.parse(self.xml_path)
            self.master_xml_root = self.master_xml_tree.getroot()
            self.master_xml_name = (
                "master_"
                + self.master_extra_name
                + os.path.splitext(os.path.basename(self.xml_path))[0]
                + ".xml"
            )
            self.master_xml_path = os.path.dirname(self.xml_path)

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
        self.unique_tracklet_ids = []

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
        self.imagesize = (
            int(float(self.settings.get("nframes"))),
            int(float(self.settings.get("nslices"))),
            int(float(self.settings.get("height"))),
            int(float(self.settings.get("width"))),
        )
        self.xcalibration = float(self.settings.get("pixelwidth"))
        self.ycalibration = float(self.settings.get("pixelheight"))
        self.zcalibration = float(self.settings.get("voxeldepth"))
        self.tcalibration = float(self.settings.get("timeinterval"))
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
        if self.channel_seg_image is None and self.enhanced_computation:
            self._get_cell_sizes()
            self._get_boundary_points()


        self.count = 0
        futures = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:

            for frame in self.Spotobjects.findall("SpotsInFrame"):
                futures.append(executor.submit(self._spot_computer, frame))
            if self.progress_bar is not None:

                self.progress_bar.label = "Collecting Spots"
                self.progress_bar.range = (
                    0,
                    len(futures),
                )
                self.progress_bar.show()

            for r in concurrent.futures.as_completed(futures):
                self.count = self.count + 1
                if self.progress_bar is not None:
                    self.progress_bar.value = self.count
                r.result()
        if self.channel_seg_image is None and self.enhanced_computation:
            self._correct_track_status()
        print(f"Iterating over tracks {len(self.filtered_track_ids)}")
        self.count = 0
        if self.progress_bar is not None:
            self.progress_bar.label = "Collecting Tracks"
            self.progress_bar.range = (0, len(self.filtered_track_ids))
            self.progress_bar.show()
        if self.channel_seg_image is None:
            for track in self.tracks.findall("Track"):
                track_id = int(track.get(self.trackid_key))

            for track in tqdm(self.tracks.findall("Track")):
                track_id = int(track.get(self.trackid_key))
                if track_id in self.filtered_track_ids:
                    self._track_computer(track, track_id)
                    self.count += 1
                    if self.progress_bar is not None:
                        self.progress_bar.value = self.count
        if self.channel_seg_image is not None:
            self._create_second_channel_xml()

        for (k, v) in self.graph_split.items():

            daughter_track_id = int(
                float(
                    str(self.unique_spot_properties[int(float(k))][self.uniqueid_key])
                )
            )
            parent_track_id = int(
                float(
                    str(self.unique_spot_properties[int(float(v))][self.uniqueid_key])
                )
            )
            self.graph_tracks[daughter_track_id] = parent_track_id

        self._get_cell_fate_tracks()
        if (
            self.spot_csv_path is not None
            and self.track_csv_path is not None
            and self.edges_csv_path is not None
        ):
            self._get_attributes()
        if (
            self.autoencoder_model or self.enhance_trackmate_xml
        ) and self.seg_image is not None:

            self._assign_cluster_class()

            self._create_master_xml()
        self.count = 0
        for index, track_id in enumerate(self.filtered_track_ids):
            if self.progress_bar is not None:
                self.progress_bar.label = "Just one more thing"
                self.progress_bar.range = (
                    0,
                    len(self.filtered_track_ids),
                )
                self.progress_bar.show()
                self.count = self.count + 1
                self.progress_bar.value = self.count
            track = self.filtered_tracks[index]
            if self.channel_seg_image is None:
                self._final_tracks(track_id)
        if self.channel_seg_image is None:
            self._compute_phenotypes()
            self._temporal_plots_trackmate()

    def _correct_track_status(self):
        if self.oneat_csv_file is not None:
            self.count = 0

            detections = pd.read_csv(self.oneat_csv_file, delimiter=",")
            cutoff_score = self.oneat_threshold_cutoff
            filtered_detections = detections[detections["Score"] > cutoff_score]
            if self.progress_bar is not None:
                self.progress_bar.label = "Oneating tracks"
                self.progress_bar.range = (
                    0,
                    len(filtered_detections) + 1,
                )

                self.progress_bar.value = self.count
                self.progress_bar.show()

            for index, row in filtered_detections.iterrows():
                t = int(row["T"])
                z = round(row["Z"])
                y = round(row["Y"])
                x = round(row["X"])
                spot = (t, z, y, x)

                self.count += 1

                spot_id = find_closest_key(spot, self.unique_oneat_spot_centroid, 0, 5)
                if spot_id is not None:
                    self.oneat_dividing_tracks[spot_id] = spot

                if self.progress_bar is not None:
                    self.progress_bar.value = self.count

    def _get_cell_fate_tracks(self):

        if self.goblet_csv_file is not None:
            print("Reading Goblet location file")
            self.goblet_dataframe = pd.read_csv(self.goblet_csv_file)
            self.GobletTrackIds.extend(
                self._get_trackmate_ids_by_location(self.goblet_dataframe)
            )
            self._update_spot_fate(self.GobletTrackIds, self.goblet_label)
        if self.basal_csv_file is not None:
            print("Reading Basal location file")
            self.basal_dataframe = pd.read_csv(self.basal_csv_file)
            self.BasalTrackIds.extend(
                self._get_trackmate_ids_by_location(self.basal_dataframe)
            )
            self._update_spot_fate(self.BasalTrackIds, self.basal_label)
        if self.radial_csv_file is not None:
            print("Reading Radial location file")
            self.radial_dataframe = pd.read_csv(self.radial_csv_file)
            self.RadialTrackIds.extend(
                self._get_trackmate_ids_by_location(self.radial_dataframe)
            )
            self._update_spot_fate(self.RadialTrackIds, self.radial_label)

    def _update_spot_fate(self, TrackIds, fate_label):

        for track_id in TrackIds:
            cell_ids = None
            self.unique_track_fate_label[track_id] = fate_label
            if track_id is not None:
                if track_id is not self.TrackidBox:
                    cell_ids = self.all_current_cell_ids[int(track_id)]
            if cell_ids is not None:
                for cell_id in cell_ids:
                    self.unique_spot_properties[cell_id].update(
                        {self.fate_key: fate_label}
                    )

    def _get_trackmate_ids_by_location(
        self, dataframe: pd.DataFrame, tracklet_length=None
    ):
        trackmate_track_ids = []
        t = int(self.tend)
        dataframe.columns = map(str.lower, dataframe.columns)
        for index, row in dataframe.iterrows():
            if "axis-0" in row:
                z = round(row["axis-0"])
                y = round(row["axis-1"])
                x = round(row["axis-2"])
                spot = (t, z, y, x)

            if "z" in row:
                t = int(round(row["t"]))
                z = round(row["z"])
                y = round(row["y"])
                x = round(row["x"])
                spot = (t, z, y, x)

            spot_id = find_closest_key(
                spot, self.unique_oneat_spot_centroid, self.time_veto, self.space_veto
            )
            if spot_id is not None:
                spot_properties_dict = self.unique_spot_properties[spot_id]
                if self.trackid_key in spot_properties_dict.keys():
                    trackmate_track_id = spot_properties_dict[self.trackid_key]
                    trackmate_track_ids.append(trackmate_track_id)

        return trackmate_track_ids

    def _create_master_xml(self):

        for Spotobject in self.master_xml_root.iter("Spot"):
            cell_id = int(Spotobject.get(self.spotid_key))
            if cell_id in self.unique_spot_properties.keys():

                for k in self.unique_spot_properties[cell_id].keys():

                    Spotobject.set(k, str(self.unique_spot_properties[cell_id][k]))

        self.master_xml_tree.write(
            os.path.join(self.master_xml_path, self.master_xml_name)
        )

    def _compute_latent_space(self):

        self.axes = self.axes.replace("T", "")
        latent_feature_list = [-1] * self.latent_features
        for (k, v) in self.unique_spot_properties.items():
            self.unique_spot_properties[k].update(
                {self.latent_shape_features_key: latent_feature_list}
            )
        for count, time_key in tqdm(
            enumerate(self._timed_centroid.keys()),
            desc="Extracting Latent Features",
            unit="_time_frame",
        ):
            tree, spot_centroids = self._timed_centroid[time_key]
            if self.progress_bar is not None:
                self.progress_bar.label = "Autoencoder for latent shape features"
                self.progress_bar.range = (
                    0,
                    len(self._timed_centroid.keys()) + 1,
                )
                self.progress_bar.value = count
                self.progress_bar.show()

            cluster_eval = Clustering(
                self.pretrainer,
                self.accelerator,
                self.devices,
                self.seg_image[int(time_key), :],
                self.axes,
                self.num_points,
                self.autoencoder_model,
                key=time_key,
                progress_bar=self.progress_bar,
                batch_size=self.batch_size,
                scale_z=self.scale_z,
                scale_xy=self.scale_xy,
                center=self.center,
                compute_with_autoencoder=False,
            )

            cluster_eval._compute_latent_features()

            (
                timed_latent_features,
                output_cluster_centroids,
                output_eigenvalues,
            ) = cluster_eval.timed_latent_features[time_key]

            for i in range(len(timed_latent_features)):
                latent_feature_list = timed_latent_features[i]
                centroid = output_cluster_centroids[i]
                quality = math.pow(
                    output_eigenvalues[i][0]
                    * output_eigenvalues[i][1]
                    * output_eigenvalues[i][2],
                    1.0 / 3.0,
                )
                dist, index = tree.query(centroid)

                if dist < quality:
                    closest_centroid = spot_centroids[index]
                    frame_spot_centroid = (
                        int(time_key),
                        closest_centroid[0],
                        closest_centroid[1],
                        closest_centroid[2],
                    )
                    closest_cell_id = self.unique_spot_centroid[frame_spot_centroid]
                    if (
                        self.unique_spot_properties[int(closest_cell_id)][
                            self.radius_key
                        ]
                        > 0
                    ):
                        self.unique_spot_properties[int(closest_cell_id)].update(
                            {self.latent_shape_features_key: latent_feature_list}
                        )
            for (k, v) in self.root_spots.items():
                self.root_spots[k] = self.unique_spot_properties[k]

    def _assign_cluster_class(self):

        self.axes = self.axes.replace("T", "")

        for count, time_key in tqdm(
            enumerate(self._timed_centroid.keys()),
            desc="Getting point clouds",
            unit="_time_frame",
        ):

            tree, spot_centroids = self._timed_centroid[time_key]
            if self.progress_bar is not None:
                self.progress_bar.label = "Autoencoder for refining point clouds"
                self.progress_bar.range = (
                    0,
                    len(self._timed_centroid.keys()) + 1,
                )
                self.progress_bar.value = count
                self.progress_bar.show()

            cluster_eval = Clustering(
                self.pretrainer,
                self.accelerator,
                self.devices,
                self.seg_image[int(time_key), :],
                self.axes,
                self.num_points,
                self.autoencoder_model,
                key=time_key,
                progress_bar=self.progress_bar,
                batch_size=self.batch_size,
                scale_z=self.scale_z,
                scale_xy=self.scale_xy,
                center=self.center,
                compute_with_autoencoder=self.compute_with_autoencoder,
            )
            cluster_eval._create_cluster_labels()

            timed_cluster_label = cluster_eval.timed_cluster_label
            (
                output_labels,
                output_cluster_centroid,
                output_cloud_eccentricity,
                output_eigenvectors,
                output_eigenvalues,
                output_dimensions,
                output_cloud_surface_area,
            ) = timed_cluster_label[time_key]
            scale_1 = 1
            scale_2 = 1
            scale_3 = 1
            for i in range(len(output_cluster_centroid)):
                centroid = output_cluster_centroid[i]
                if not isinstance(output_eigenvalues[i], int):
                    quality = math.pow(
                        output_eigenvalues[i][2]
                        * output_eigenvalues[i][1]
                        * output_eigenvalues[i][0],
                        1.0 / 3.0,
                    )
                    eccentricity_comp_firstyz = output_cloud_eccentricity[i]
                    eccentricity_dimension = output_dimensions[i]
                    if not isinstance(eccentricity_dimension, int):

                        scale_1 = self.xcalibration
                        scale_2 = self.ycalibration
                        scale_3 = self.zcalibration

                        cell_axis_x = output_eigenvectors[i][2]
                        cell_axis_y = output_eigenvectors[i][1]
                        cell_axis_z = output_eigenvectors[i][0]

                        surface_area = (
                            output_cloud_surface_area[i]
                            * self.zcalibration
                            * self.ycalibration
                            * self.xcalibration
                        )
                        dist, index = tree.query(centroid)
                        radius = quality * math.pow(
                            self.zcalibration * self.xcalibration * self.ycalibration,
                            1.0 / 3.0,
                        )
                        if dist < quality:
                            closest_centroid = spot_centroids[index]
                            frame_spot_centroid = (
                                int(time_key),
                                closest_centroid[0],
                                closest_centroid[1],
                                closest_centroid[2],
                            )
                            closest_cell_id = self.unique_spot_centroid[
                                frame_spot_centroid
                            ]

                            angle_cell_axis_x = cell_angular_change_x(cell_axis_x)
                            angle_cell_axis_y = cell_angular_change_y(cell_axis_y)
                            angle_cell_axis_z = cell_angular_change_z(cell_axis_z)

                            self.unique_spot_properties[int(closest_cell_id)].update(
                                {self.cell_axis_x_key: angle_cell_axis_x}
                            )
                            self.unique_spot_properties[int(closest_cell_id)].update(
                                {self.cell_axis_y_key: angle_cell_axis_y}
                            )
                            self.unique_spot_properties[int(closest_cell_id)].update(
                                {self.cell_axis_z_key: angle_cell_axis_z}
                            )
                            if (
                                self.unique_spot_properties[int(closest_cell_id)][
                                    self.radius_key
                                ]
                                > 0
                            ):
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update(
                                    {
                                        self.eccentricity_comp_firstkey: eccentricity_comp_firstyz[
                                            2
                                        ]
                                        * scale_1
                                    }
                                )
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update(
                                    {
                                        self.eccentricity_comp_secondkey: eccentricity_comp_firstyz[
                                            1
                                        ]
                                        * scale_2
                                    }
                                )

                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update(
                                    {
                                        self.eccentricity_comp_thirdkey: eccentricity_comp_firstyz[
                                            0
                                        ]
                                        * scale_3
                                    }
                                )

                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.surface_area_key: surface_area})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.quality_key: quality})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.radius_key: radius})
                            else:

                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.eccentricity_comp_firstkey: -1})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.eccentricity_comp_secondkey: -1})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.eccentricity_comp_thirdkey: -1})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.surface_area_key: -1})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.quality_key: -1})
                                self.unique_spot_properties[
                                    int(closest_cell_id)
                                ].update({self.radius_key: -1})

            for (k, v) in self.root_spots.items():
                self.root_spots[k] = self.unique_spot_properties[k]

    def _compute_phenotypes(self):

        for (k, v) in self.unique_tracks.items():

            track_id = k
            tracklet_properties = self.unique_track_properties[k]
            tracks = self.unique_tracks[k]

            Z = tracks[:, 2]
            Y = tracks[:, 3]
            X = tracks[:, 4]

            time = tracklet_properties[:, 0]
            unique_ids = tracklet_properties[:, 1]
            unique_ids_set = set(unique_ids)
            # generation_ids = tracklet_properties[:, 2]
            radius = tracklet_properties[:, 3]
            eccentricity_comp_first = tracklet_properties[:, 4]
            eccentricity_comp_second = tracklet_properties[:, 5]
            eccentricity_comp_third = tracklet_properties[:, 6]
            surface_area = tracklet_properties[:, 7]

            intensity = tracklet_properties[:, 8]
            speed = tracklet_properties[:, 9]

            motion_angle_z = tracklet_properties[:, 10]
            motion_angle_y = tracklet_properties[:, 11]
            motion_angle_x = tracklet_properties[:, 12]

            acceleration = tracklet_properties[:, 13]
            distance_cell_mask = tracklet_properties[:, 14]
            local_density = tracklet_properties[:, 15]

            radial_angle_z = tracklet_properties[:, 16]
            radial_angle_y = tracklet_properties[:, 17]
            radial_angle_x = tracklet_properties[:, 18]

            cell_axis_z = tracklet_properties[:, 19]
            cell_axis_y = tracklet_properties[:, 20]
            cell_axis_x = tracklet_properties[:, 21]

            track_displacement = tracklet_properties[:, 22]

            total_track_distance = tracklet_properties[:, 23]

            max_track_distance = tracklet_properties[:, 24]

            track_duration = tracklet_properties[:, 25]

            msd = tracklet_properties[:, 26]

            if tracklet_properties.shape[1] > 26:
                latent_shape_features = tracklet_properties[:, 27:]
            else:
                latent_shape_features = []

            unique_fft_properties_tracklet = {}
            unique_cluster_properties_tracklet = {}
            self.unique_fft_properties[track_id] = {}
            self.unique_cluster_properties[track_id] = {}

            unique_shape_properties_tracklet = {}
            unique_dynamic_properties_tracklet = {}
            self.unique_shape_properties[track_id] = {}
            self.unique_dynamic_properties[track_id] = {}
            expanded_time = np.zeros(self.tend - self.tstart + 1)
            point_sample = expanded_time.shape[0]
            for i in range(len(expanded_time)):
                expanded_time[i] = i
            for current_unique_id in unique_ids_set:


                current_time = []
                current_z = []
                current_y = []
                current_x = []
                current_intensity = []
                current_radius = []
                current_speed = []
                current_motion_angle_z = []
                current_motion_angle_y = []
                current_motion_angle_x = []

                current_acceleration = []
                current_distance_cell_mask = []
                current_local_density = []
                current_eccentricity_comp_first = []
                current_eccentricity_comp_second = []
                current_eccentricity_comp_third = []
                current_surface_area = []
                current_latent_shape_features = []
                current_radial_angle_z = []
                current_radial_angle_y = []
                current_radial_angle_x = []
                current_cell_axis_z = []
                current_cell_axis_y = []
                current_cell_axis_x = []
                current_track_displacement = []
                current_total_track_distance = []
                current_max_track_distance = []
                current_track_duration = []
                current_msd = []

                for j in range(time.shape[0]):
                    if current_unique_id == unique_ids[j]:
                        current_time.append(time[j])
                        current_z.append(Z[j])
                        current_y.append(Y[j])
                        current_x.append(X[j])
                        current_intensity.append(intensity[j])
                        current_radius.append(radius[j])
                        current_speed.append(speed[j])
                        current_motion_angle_z.append(motion_angle_z[j])
                        current_motion_angle_y.append(motion_angle_y[j])
                        current_motion_angle_x.append(motion_angle_x[j])

                        current_acceleration.append(acceleration[j])
                        current_distance_cell_mask.append(distance_cell_mask[j])
                        current_local_density.append(local_density[j])
                        current_eccentricity_comp_first.append(
                            eccentricity_comp_first[j]
                        )
                        current_eccentricity_comp_second.append(
                            eccentricity_comp_second[j]
                        )
                        current_eccentricity_comp_third.append(
                            eccentricity_comp_third[j]
                        )

                        current_surface_area.append(surface_area[j])
                        if latent_shape_features != []:
                            current_latent_shape_features.append(
                                latent_shape_features[j]
                            )
                        current_radial_angle_z.append(radial_angle_z[j])
                        current_radial_angle_y.append(radial_angle_y[j])
                        current_radial_angle_x.append(radial_angle_x[j])

                        current_cell_axis_z.append(cell_axis_z[j])
                        current_cell_axis_y.append(cell_axis_y[j])
                        current_cell_axis_x.append(cell_axis_x[j])

                        current_track_displacement.append(track_displacement[j])
                        current_total_track_distance.append(total_track_distance[j])
                        current_max_track_distance.append(max_track_distance[j])
                        current_track_duration.append(track_duration[j])
                        current_msd.append(msd[j])

                current_time = np.asarray(current_time, dtype=np.float32)
                current_intensity = np.asarray(current_intensity, dtype=np.float32)

                current_radius = np.asarray(current_radius, dtype=np.float32)

                current_eccentricity_comp_first = np.asarray(
                    current_eccentricity_comp_first, dtype=np.float32
                )
                current_eccentricity_comp_second = np.asarray(
                    current_eccentricity_comp_second, dtype=np.float32
                )
                current_eccentricity_comp_third = np.asarray(
                    current_eccentricity_comp_third, dtype=np.float32
                )
                current_surface_area = np.asarray(
                    current_surface_area, dtype=np.float32
                )

                current_latent_shape_features = np.asarray(
                    current_latent_shape_features, dtype=np.float32
                )

                current_speed = np.asarray(current_speed, dtype=np.float32)
                current_motion_angle_z = np.asarray(
                    current_motion_angle_z, dtype=np.float32
                )
                current_motion_angle_y = np.asarray(
                    current_motion_angle_y, dtype=np.float32
                )
                current_motion_angle_x = np.asarray(
                    current_motion_angle_x, dtype=np.float32
                )
                current_acceleration = np.asarray(
                    current_acceleration, dtype=np.float32
                )
                current_distance_cell_mask = np.asarray(
                    current_distance_cell_mask, dtype=np.float32
                )

                current_local_density = np.asarray(
                    current_local_density, dtype=np.float32
                )
                current_radial_angle_z = np.asarray(
                    current_radial_angle_z, dtype=np.float32
                )
                current_radial_angle_y = np.asarray(
                    current_radial_angle_y, dtype=np.float32
                )
                current_radial_angle_x = np.asarray(
                    current_radial_angle_x, dtype=np.float32
                )

                current_cell_axis_z = np.asarray(current_cell_axis_z, dtype=np.float32)
                current_cell_axis_y = np.asarray(current_cell_axis_y, dtype=np.float32)
                current_cell_axis_x = np.asarray(current_cell_axis_x, dtype=np.float32)

                current_track_displacement = np.asarray(
                    current_track_displacement, dtype=np.float32
                )
                current_total_track_distance = np.asarray(
                    current_total_track_distance, dtype=np.float32
                )
                current_max_track_distance = np.asarray(
                    current_max_track_distance, dtype=np.float32
                )
                current_track_duration = np.asarray(
                    current_track_duration, dtype=np.float32
                )
                current_msd = np.asarray(current_msd, dtype=np.float32)

                if point_sample > 0:
                    xf_sample = fftfreq(point_sample, self.tcalibration)
                    fftstrip_sample = fft(current_intensity)
                    ffttotal_sample = np.abs(fftstrip_sample)
                    xf_sample = xf_sample[0 : len(xf_sample) // 2]
                    ffttotal_sample = ffttotal_sample[0 : len(ffttotal_sample) // 2]

                unique_fft_properties_tracklet[current_unique_id] = (
                    expanded_time,
                    current_intensity,
                    xf_sample,
                    ffttotal_sample,
                )
                unique_cluster_properties_tracklet[current_unique_id] = current_time
                unique_shape_properties_tracklet[current_unique_id] = (
                    current_time,
                    current_z,
                    current_y,
                    current_x,
                    current_radius,
                    current_eccentricity_comp_first,
                    current_eccentricity_comp_second,
                    current_eccentricity_comp_third,
                    current_local_density,
                    current_surface_area,
                    current_latent_shape_features,
                )
                unique_dynamic_properties_tracklet[current_unique_id] = (
                    current_time,
                    current_speed,
                    current_motion_angle_z,
                    current_motion_angle_y,
                    current_motion_angle_x,
                    current_acceleration,
                    current_distance_cell_mask,
                    current_radial_angle_z,
                    current_radial_angle_y,
                    current_radial_angle_x,
                    current_cell_axis_z,
                    current_cell_axis_y,
                    current_cell_axis_x,
                    current_track_displacement,
                    current_total_track_distance,
                    current_max_track_distance,
                    current_track_duration,
                    current_msd,
                )
                self.unique_fft_properties[track_id].update(
                    {
                        current_unique_id: unique_fft_properties_tracklet[
                            current_unique_id
                        ]
                    }
                )
                self.unique_cluster_properties[track_id].update(
                    {
                        current_unique_id: unique_cluster_properties_tracklet[
                            current_unique_id
                        ]
                    }
                )

                self.unique_shape_properties[track_id].update(
                    {
                        current_unique_id: unique_shape_properties_tracklet[
                            current_unique_id
                        ]
                    }
                )
                self.unique_dynamic_properties[track_id].update(
                    {
                        current_unique_id: unique_dynamic_properties_tracklet[
                            current_unique_id
                        ]
                    }
                )

    def _second_channel_spots(self, frame, z, y, x, cell_id, track_id):

        (
            tree,
            centroids,
            labels,
            volume,
            intensity_mean,
            intensity_total,
            bounding_boxes,
        ) = self._timed_channel_seg_image[str(int(float(frame)))]
        pixeltestlocation = (z, y, x)
        dist, index = tree.query(pixeltestlocation)

        bbox = bounding_boxes[index]
        sizez = abs(bbox[0] - bbox[3])
        sizey = abs(bbox[1] - bbox[4])
        sizex = abs(bbox[2] - bbox[5])
        veto_volume = sizex * sizey * sizez
        veto_radius = math.pow(3 * veto_volume / (4 * math.pi), 1.0 / 3.0)
        location = (
            centroids[index][0] * self.zcalibration,
            centroids[index][1] * self.ycalibration,
            centroids[index][2] * self.xcalibration,
        )
        QUALITY = math.pow(volume[index], 1.0 / 3.0)
        RADIUS = math.pow(
            volume[index] * self.xcalibration * self.ycalibration * self.zcalibration,
            1.0 / 3.0,
        )

        distance_cell_mask, maskcentroid = self._get_boundary_dist(frame, location)
        
        if self.seg_image is not None or self.channel_seg_image is not None:
               local_density = self._get_label_density(frame, location)
        else:
               local_cell_density = 1
        if dist <= 2 * veto_radius:

            if track_id not in self.matched_second_channel_tracks:

                self.matched_second_channel_tracks.append(track_id)

                self.channel_unique_spot_properties[cell_id] = {
                    self.cellid_key: int(cell_id),
                    self.frameid_key: int(frame),
                    self.zposid_key: float(centroids[index][0] * self.zcalibration),
                    self.yposid_key: float(centroids[index][1] * self.ycalibration),
                    self.xposid_key: float(centroids[index][2] * self.xcalibration),
                    self.trackid_key: int(track_id),
                    self.total_intensity_key: (float(intensity_total[index])),
                    self.mean_intensity_key: (float(intensity_mean[index])),
                    self.radius_key: (float(RADIUS)),
                    self.quality_key: (float(QUALITY)),
                    self.distance_cell_mask_key: float(distance_cell_mask),
                    self.local_cell_density_key: float(local_density),
                    self.maskcentroid_z_key: float(maskcentroid[0]),
                    self.maskcentroid_y_key: float(maskcentroid[1]),
                    self.maskcentroid_x_key: float(maskcentroid[2]),
                }
            else:
                self.channel_unique_spot_properties[
                    cell_id
                ] = self.unique_spot_properties[cell_id]
                self.channel_unique_spot_properties[cell_id].update(
                    {self.total_intensity_key: -1}
                )
                self.channel_unique_spot_properties[cell_id].update(
                    {self.mean_intensity_key: -1}
                )
                self.channel_unique_spot_properties[cell_id].update(
                    {self.radius_key: -1}
                )
                self.channel_unique_spot_properties[cell_id].update(
                    {self.quality_key: -1}
                )

    def _get_cal(self, frame):

        frame = int(float(frame))

        if self.variable_t_calibration is not None:

            for key_time in sorted(self.variable_t_calibration.keys()):
                key_time = int(float(key_time))
                if frame <= key_time:
                    self.tcalibration = float(self.variable_t_calibration[key_time])
                    return

    def _dict_update(
        self,
        cell_id: int,
        track_id: int,
        source_id: int,
        target_id: int,
    ):
        generation_id = self.generation_dict[cell_id]
        tracklet_id = self.tracklet_dict[cell_id]

        unique_id = str(track_id) + str(generation_id) + str(tracklet_id)

        vec_cell = [
            float(self.unique_spot_properties[int(cell_id)][self.xposid_key]),
            float(self.unique_spot_properties[int(cell_id)][self.yposid_key]),
            float(self.unique_spot_properties[int(cell_id)][self.zposid_key]),
        ]

        angle_x = cell_angular_change_x(vec_cell)
        angle_y = cell_angular_change_y(vec_cell)
        angle_z = cell_angular_change_z(vec_cell)

        self.unique_spot_properties[int(cell_id)].update(
            {self.radial_angle_x_key: angle_x}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.radial_angle_y_key: angle_y}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.radial_angle_z_key: angle_z}
        )
        self.unique_tracklet_ids.append(str(unique_id))
        self.unique_spot_properties[int(cell_id)].update(
            {self.uniqueid_key: str(unique_id)}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.trackletid_key: str(tracklet_id)}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.generationid_key: str(generation_id)}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.trackid_key: str(track_id)}
        )
        self.unique_spot_properties[int(cell_id)].update({self.motion_angle_z_key: 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.motion_angle_y_key: 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.motion_angle_x_key: 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.speed_key: 0.0})
        self.unique_spot_properties[int(cell_id)].update({self.acceleration_key: 0.0})
        self.unique_spot_properties[int(cell_id)].update(
            {self.eccentricity_comp_firstkey: -1}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.eccentricity_comp_secondkey: -1}
        )
        self.unique_spot_properties[int(cell_id)].update(
            {self.eccentricity_comp_thirdkey: -1}
        )
        self.unique_spot_properties[int(cell_id)].update({self.surface_area_key: -1})
        self.unique_spot_properties[int(cell_id)].update({self.cell_axis_z_key: -1})
        self.unique_spot_properties[int(cell_id)].update({self.cell_axis_y_key: -1})
        self.unique_spot_properties[int(cell_id)].update({self.cell_axis_x_key: -1})
        if source_id is not None:
            self.unique_spot_properties[int(cell_id)].update(
                {self.beforeid_key: int(source_id)}
            )
            vec_1 = [
                float(self.unique_spot_properties[int(cell_id)][self.xposid_key])
                - float(self.unique_spot_properties[int(source_id)][self.xposid_key]),
                float(self.unique_spot_properties[int(cell_id)][self.yposid_key])
                - float(self.unique_spot_properties[int(source_id)][self.yposid_key]),
                float(self.unique_spot_properties[int(cell_id)][self.zposid_key])
                - float(self.unique_spot_properties[int(source_id)][self.zposid_key]),
            ]

            time_vec_1 = max(
                1,
                abs(
                    int(
                        float(
                            self.unique_spot_properties[int(cell_id)][self.frameid_key]
                        )
                        - float(
                            self.unique_spot_properties[int(source_id)][
                                self.frameid_key
                            ]
                        )
                    )
                ),
            )
            frame = int(
                float(self.unique_spot_properties[int(cell_id)][self.frameid_key])
            )
            self._get_cal(frame)

            speed = np.sqrt(np.dot(vec_1, vec_1)) / (time_vec_1 * self.tcalibration)
            self.unique_spot_properties[int(cell_id)].update({self.speed_key: speed})

            motion_angle_x = cell_angular_change_x(vec_1)
            motion_angle_y = cell_angular_change_y(vec_1)
            motion_angle_z = cell_angular_change_z(vec_1)

            self.unique_spot_properties[int(cell_id)].update(
                {self.motion_angle_x_key: motion_angle_x}
            )

            self.unique_spot_properties[int(cell_id)].update(
                {self.motion_angle_y_key: motion_angle_y}
            )

            self.unique_spot_properties[int(cell_id)].update(
                {self.motion_angle_z_key: motion_angle_z}
            )

            if source_id in self.edge_source_lookup:
                pre_source_id = self.edge_source_lookup[source_id]

                vec_2 = [
                    float(self.unique_spot_properties[int(cell_id)][self.xposid_key])
                    - 2
                    * float(
                        self.unique_spot_properties[int(source_id)][self.xposid_key]
                    )
                    + float(
                        self.unique_spot_properties[int(pre_source_id)][self.xposid_key]
                    ),
                    float(self.unique_spot_properties[int(cell_id)][self.yposid_key])
                    - 2
                    * float(
                        self.unique_spot_properties[int(source_id)][self.yposid_key]
                    )
                    + float(
                        self.unique_spot_properties[int(pre_source_id)][self.yposid_key]
                    ),
                    float(self.unique_spot_properties[int(cell_id)][self.zposid_key])
                    - 2
                    * float(
                        self.unique_spot_properties[int(source_id)][self.zposid_key]
                    )
                    + float(
                        self.unique_spot_properties[int(pre_source_id)][self.zposid_key]
                    ),
                ]

                time_vec_2 = max(
                    1,
                    abs(
                        int(
                            float(
                                self.unique_spot_properties[int(cell_id)][
                                    self.frameid_key
                                ]
                            )
                            - float(
                                self.unique_spot_properties[int(pre_source_id)][
                                    self.frameid_key
                                ]
                            )
                        )
                    ),
                )

                acc = np.sqrt(np.dot(vec_2, vec_2)) / (time_vec_2 * self.tcalibration)

                self.unique_spot_properties[int(cell_id)].update(
                    {self.acceleration_key: acc}
                )
        elif source_id is None:
            self.unique_spot_properties[int(cell_id)].update({self.beforeid_key: None})

        if target_id is not None:
            self.unique_spot_properties[int(cell_id)].update(
                {self.afterid_key: int(target_id)}
            )
        elif target_id is None:
            self.unique_spot_properties[int(cell_id)].update({self.afterid_key: None})

        self._second_channel_update(cell_id, track_id)

    def _msd_update(
        self,
        root_id: int,
        cell_id: int,
    ):

        vec_root = [
            float(self.unique_spot_properties[int(cell_id)][self.xposid_key])
            - float(self.unique_spot_properties[int(root_id)][self.xposid_key]),
            float(self.unique_spot_properties[int(cell_id)][self.yposid_key])
            - float(self.unique_spot_properties[int(root_id)][self.yposid_key]),
            float(self.unique_spot_properties[int(cell_id)][self.zposid_key])
            - float(self.unique_spot_properties[int(root_id)][self.zposid_key]),
        ]

        msd = np.dot(vec_root, vec_root)

        self.unique_spot_properties[int(cell_id)].update({self.msd_key: msd})

    def _temporal_plots_trackmate(self):

        self.Attr = {}
        starttime = int(self.tstart)
        endtime = int(self.tend)

        self.time = []
        self.mitotic_mean_disp_z = []
        self.mitotic_var_disp_z = []

        self.mitotic_mean_disp_y = []
        self.mitotic_var_disp_y = []

        self.mitotic_mean_disp_x = []
        self.mitotic_var_disp_x = []

        self.mitotic_mean_radius = []
        self.mitotic_var_radius = []

        self.mitotic_mean_speed = []
        self.mitotic_var_speed = []

        self.mitotic_mean_acc = []
        self.mitotic_var_acc = []

        self.mitotic_mean_directional_change_z = []
        self.mitotic_var_directional_change_z = []

        self.mitotic_mean_directional_change_y = []
        self.mitotic_var_directional_change_y = []

        self.mitotic_mean_directional_change_x = []
        self.mitotic_var_directional_change_x = []

        self.mitotic_mean_distance_cell_mask = []
        self.mitotic_var_distance_cell_mask = []

        self.mitotic_mean_local_cell_density = []
        self.mitotic_var_local_cell_density = []

        self.non_mitotic_mean_disp_z = []
        self.non_mitotic_var_disp_z = []

        self.non_mitotic_mean_disp_y = []
        self.non_mitotic_var_disp_y = []

        self.non_mitotic_mean_disp_x = []
        self.non_mitotic_var_disp_x = []

        self.non_mitotic_mean_radius = []
        self.non_mitotic_var_radius = []

        self.non_mitotic_mean_speed = []
        self.non_mitotic_var_speed = []

        self.non_mitotic_mean_acc = []
        self.non_mitotic_var_acc = []

        self.non_mitotic_mean_directional_change_z = []
        self.non_mitotic_var_directional_change_z = []

        self.non_mitotic_mean_directional_change_y = []
        self.non_mitotic_var_directional_change_y = []

        self.non_mitotic_mean_directional_change_x = []
        self.non_mitotic_var_directional_change_x = []

        self.non_mitotic_mean_distance_cell_mask = []
        self.non_mitotic_var_distance_cell_mask = []

        self.non_mitotic_mean_local_cell_density = []
        self.non_mitotic_var_local_cell_density = []

        self.all_mean_disp_z = []
        self.all_var_disp_z = []

        self.all_mean_disp_y = []
        self.all_var_disp_y = []

        self.all_mean_disp_x = []
        self.all_var_disp_x = []

        self.all_mean_radius = []
        self.all_var_radius = []

        self.all_mean_speed = []
        self.all_var_speed = []

        self.all_mean_acc = []
        self.all_var_acc = []

        self.all_mean_directional_change_z = []
        self.all_var_directional_change_z = []

        self.all_mean_directional_change_y = []
        self.all_var_directional_change_y = []

        self.all_mean_directional_change_x = []
        self.all_var_directional_change_x = []

        self.all_mean_distance_cell_mask = []
        self.all_var_distance_cell_mask = []

        self.all_mean_distance_cell_mask = []
        self.all_var_distance_cell_mask = []

        self.all_mean_local_cell_density = []
        self.all_var_local_cell_density = []

        self.all_mean_local_cell_density = []
        self.all_var_local_cell_density = []

        self.goblet_mean_disp_z = []
        self.goblet_var_disp_z = []

        self.goblet_mean_disp_y = []
        self.goblet_var_disp_y = []

        self.goblet_mean_disp_x = []
        self.goblet_var_disp_x = []

        self.goblet_mean_radius = []
        self.goblet_var_radius = []

        self.goblet_mean_speed = []
        self.goblet_var_speed = []

        self.goblet_mean_acc = []
        self.goblet_var_acc = []

        self.goblet_mean_directional_change_z = []
        self.goblet_var_directional_change_z = []

        self.goblet_mean_directional_change_y = []
        self.goblet_var_directional_change_y = []

        self.goblet_mean_directional_change_x = []
        self.goblet_var_directional_change_x = []

        self.goblet_mean_distance_cell_mask = []
        self.goblet_var_distance_cell_mask = []

        self.goblet_mean_local_cell_density = []
        self.goblet_var_local_cell_density = []

        self.basal_mean_disp_z = []
        self.basal_var_disp_z = []

        self.basal_mean_disp_y = []
        self.basal_var_disp_y = []

        self.basal_mean_disp_x = []
        self.basal_var_disp_x = []

        self.basal_mean_radius = []
        self.basal_var_radius = []

        self.basal_mean_speed = []
        self.basal_var_speed = []

        self.basal_mean_acc = []
        self.basal_var_acc = []

        self.basal_mean_directional_change_z = []
        self.basal_var_directional_change_z = []

        self.basal_mean_directional_change_y = []
        self.basal_var_directional_change_y = []

        self.basal_mean_directional_change_x = []
        self.basal_var_directional_change_x = []

        self.basal_mean_distance_cell_mask = []
        self.basal_var_distance_cell_mask = []

        self.basal_mean_local_cell_density = []
        self.basal_var_local_cell_density = []

        self.radial_mean_disp_z = []
        self.radial_var_disp_z = []

        self.radial_mean_disp_y = []
        self.radial_var_disp_y = []

        self.radial_mean_disp_x = []
        self.radial_var_disp_x = []

        self.radial_mean_radius = []
        self.radial_var_radius = []

        self.radial_mean_speed = []
        self.radial_var_speed = []

        self.radial_mean_acc = []
        self.radial_var_acc = []

        self.radial_mean_directional_change_z = []
        self.radial_var_directional_change_z = []

        self.radial_mean_directional_change_y = []
        self.radial_var_directional_change_y = []

        self.radial_mean_directional_change_x = []
        self.radial_var_directional_change_x = []

        self.radial_mean_distance_cell_mask = []
        self.radial_var_distance_cell_mask = []

        self.radial_mean_local_cell_density = []
        self.radial_var_local_cell_density = []

        all_spots_tracks = {}
        for (k, v) in self.unique_spot_properties.items():

            all_spots = self.unique_spot_properties[k]
            if self.trackid_key in all_spots:
                all_spots_tracks[k] = all_spots

        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:

            for i in tqdm(range(starttime, endtime), total=endtime - starttime):

                futures.append(
                    executor.submit(self._compute_temporal, i, all_spots_tracks)
                )

            [r.result() for r in concurrent.futures.as_completed(futures)]

    def _compute_temporal(self, i, all_spots_tracks):
        mitotic_disp_z = []
        mitotic_disp_y = []
        mitotic_disp_x = []
        mitotic_radius = []
        mitotic_speed = []
        mitotic_acc = []
        mitotic_directional_change_z = []
        mitotic_directional_change_y = []
        mitotic_directional_change_x = []
        mitotic_distance_cell_mask = []
        mitotic_local_cell_density = []

        non_mitotic_disp_z = []
        non_mitotic_disp_y = []
        non_mitotic_disp_x = []
        non_mitotic_radius = []
        non_mitotic_speed = []
        non_mitotic_acc = []
        non_mitotic_directional_change_z = []
        non_mitotic_directional_change_y = []
        non_mitotic_directional_change_x = []
        non_mitotic_distance_cell_mask = []
        non_mitotic_local_cell_density = []

        all_disp_z = []
        all_disp_y = []
        all_disp_x = []
        all_radius = []
        all_speed = []
        all_acc = []
        all_directional_change_z = []
        all_directional_change_y = []
        all_directional_change_x = []
        all_distance_cell_mask = []
        all_local_cell_density = []

        goblet_disp_z = []
        goblet_disp_y = []
        goblet_disp_x = []
        goblet_radius = []
        goblet_speed = []
        goblet_acc = []
        goblet_directional_change_z = []
        goblet_directional_change_y = []
        goblet_directional_change_x = []
        goblet_distance_cell_mask = []
        goblet_local_cell_density = []

        basal_disp_z = []
        basal_disp_y = []
        basal_disp_x = []
        basal_radius = []
        basal_speed = []
        basal_acc = []
        basal_directional_change_z = []
        basal_directional_change_y = []
        basal_directional_change_x = []
        basal_distance_cell_mask = []
        basal_local_cell_density = []

        radial_disp_z = []
        radial_disp_y = []
        radial_disp_x = []
        radial_radius = []
        radial_speed = []
        radial_acc = []
        radial_directional_change_z = []
        radial_directional_change_y = []
        radial_directional_change_x = []
        radial_distance_cell_mask = []
        radial_local_cell_density = []

        for (k, v) in all_spots_tracks.items():

            current_time = all_spots_tracks[k][self.frameid_key]
            mitotic = all_spots_tracks[k][self.dividing_key]
            cell_fate = all_spots_tracks[k][self.fate_key]
            if i == int(current_time):

                self._get_cal(current_time)

                if cell_fate == self.goblet_label:

                    goblet_disp_z.append(all_spots_tracks[k][self.zposid_key])
                    goblet_disp_y.append(all_spots_tracks[k][self.yposid_key])
                    goblet_disp_x.append(all_spots_tracks[k][self.xposid_key])
                    if all_spots_tracks[k][self.radius_key] > 0:
                        goblet_radius.append(all_spots_tracks[k][self.radius_key])
                    else:
                        goblet_radius.append(None)
                    goblet_speed.append(all_spots_tracks[k][self.speed_key])
                    goblet_acc.append(all_spots_tracks[k][self.acceleration_key])
                    goblet_directional_change_z.append(
                        all_spots_tracks[k][self.motion_angle_z_key]
                    )
                    goblet_directional_change_y.append(
                        all_spots_tracks[k][self.motion_angle_y_key]
                    )
                    goblet_directional_change_x.append(
                        all_spots_tracks[k][self.motion_angle_x_key]
                    )
                    goblet_distance_cell_mask.append(
                        all_spots_tracks[k][self.distance_cell_mask_key]
                    )
                    goblet_local_cell_density.append(
                        all_spots_tracks[k][self.local_cell_density_key]
                    )

                if cell_fate == self.basal_label:

                    basal_disp_z.append(all_spots_tracks[k][self.zposid_key])
                    basal_disp_y.append(all_spots_tracks[k][self.yposid_key])
                    basal_disp_x.append(all_spots_tracks[k][self.xposid_key])
                    if all_spots_tracks[k][self.radius_key] > 0:
                        basal_radius.append(all_spots_tracks[k][self.radius_key])
                    else:
                        basal_radius.append(None)
                    basal_speed.append(all_spots_tracks[k][self.speed_key])
                    basal_acc.append(all_spots_tracks[k][self.acceleration_key])
                    basal_directional_change_z.append(
                        all_spots_tracks[k][self.motion_angle_z_key]
                    )
                    basal_directional_change_y.append(
                        all_spots_tracks[k][self.motion_angle_y_key]
                    )
                    basal_directional_change_x.append(
                        all_spots_tracks[k][self.motion_angle_x_key]
                    )
                    basal_distance_cell_mask.append(
                        all_spots_tracks[k][self.distance_cell_mask_key]
                    )
                    basal_local_cell_density.append(
                        all_spots_tracks[k][self.local_cell_density_key]
                    )

                if cell_fate == self.radial_label:

                    radial_disp_z.append(all_spots_tracks[k][self.zposid_key])
                    radial_disp_y.append(all_spots_tracks[k][self.yposid_key])
                    radial_disp_x.append(all_spots_tracks[k][self.xposid_key])
                    if all_spots_tracks[k][self.radius_key] > 0:
                        radial_radius.append(all_spots_tracks[k][self.radius_key])
                    else:
                        radial_radius.append(None)
                    radial_speed.append(all_spots_tracks[k][self.speed_key])
                    radial_acc.append(all_spots_tracks[k][self.acceleration_key])
                    radial_directional_change_z.append(
                        all_spots_tracks[k][self.motion_angle_z_key]
                    )
                    radial_directional_change_y.append(
                        all_spots_tracks[k][self.motion_angle_y_key]
                    )
                    radial_directional_change_x.append(
                        all_spots_tracks[k][self.motion_angle_x_key]
                    )
                    radial_distance_cell_mask.append(
                        all_spots_tracks[k][self.distance_cell_mask_key]
                    )
                    radial_local_cell_density.append(
                        all_spots_tracks[k][self.local_cell_density_key]
                    )

                if mitotic:
                    mitotic_disp_z.append(all_spots_tracks[k][self.zposid_key])
                    mitotic_disp_y.append(all_spots_tracks[k][self.yposid_key])
                    mitotic_disp_x.append(all_spots_tracks[k][self.xposid_key])
                    if all_spots_tracks[k][self.radius_key] > 0:
                        mitotic_radius.append(all_spots_tracks[k][self.radius_key])
                    else:
                        mitotic_radius.append(None)
                    mitotic_speed.append(all_spots_tracks[k][self.speed_key])
                    mitotic_acc.append(all_spots_tracks[k][self.acceleration_key])
                    mitotic_directional_change_z.append(
                        all_spots_tracks[k][self.motion_angle_z_key]
                    )
                    mitotic_directional_change_y.append(
                        all_spots_tracks[k][self.motion_angle_y_key]
                    )
                    mitotic_directional_change_x.append(
                        all_spots_tracks[k][self.motion_angle_x_key]
                    )
                    mitotic_distance_cell_mask.append(
                        all_spots_tracks[k][self.distance_cell_mask_key]
                    )
                    mitotic_local_cell_density.append(
                        all_spots_tracks[k][self.local_cell_density_key]
                    )

                if not mitotic:
                    non_mitotic_disp_z.append(all_spots_tracks[k][self.zposid_key])
                    non_mitotic_disp_y.append(all_spots_tracks[k][self.yposid_key])
                    non_mitotic_disp_x.append(all_spots_tracks[k][self.xposid_key])
                    if all_spots_tracks[k][self.radius_key] > 0:
                        non_mitotic_radius.append(all_spots_tracks[k][self.radius_key])
                    else:
                        non_mitotic_radius.append(None)
                    non_mitotic_speed.append(all_spots_tracks[k][self.speed_key])
                    non_mitotic_acc.append(all_spots_tracks[k][self.acceleration_key])
                    non_mitotic_directional_change_z.append(
                        all_spots_tracks[k][self.motion_angle_z_key]
                    )
                    non_mitotic_directional_change_y.append(
                        all_spots_tracks[k][self.motion_angle_y_key]
                    )
                    non_mitotic_directional_change_x.append(
                        all_spots_tracks[k][self.motion_angle_x_key]
                    )
                    non_mitotic_distance_cell_mask.append(
                        all_spots_tracks[k][self.distance_cell_mask_key]
                    )
                    non_mitotic_local_cell_density.append(
                        all_spots_tracks[k][self.local_cell_density_key]
                    )

                all_disp_z.append(all_spots_tracks[k][self.zposid_key])
                all_disp_y.append(all_spots_tracks[k][self.yposid_key])
                all_disp_x.append(all_spots_tracks[k][self.xposid_key])
                if all_spots_tracks[k][self.radius_key] > 0:
                    all_radius.append(all_spots_tracks[k][self.radius_key])
                else:
                    all_radius.append(None)
                all_speed.append(all_spots_tracks[k][self.speed_key])
                all_acc.append(all_spots_tracks[k][self.acceleration_key])
                all_directional_change_z.append(
                    all_spots_tracks[k][self.motion_angle_z_key]
                )
                all_directional_change_y.append(
                    all_spots_tracks[k][self.motion_angle_y_key]
                )
                all_directional_change_x.append(
                    all_spots_tracks[k][self.motion_angle_x_key]
                )
                all_distance_cell_mask.append(
                    all_spots_tracks[k][self.distance_cell_mask_key]
                )
                all_local_cell_density.append(
                    all_spots_tracks[k][self.local_cell_density_key]
                )

        mitotic_disp_z = np.abs(np.diff(mitotic_disp_z))
        mitotic_disp_y = np.abs(np.diff(mitotic_disp_y))
        mitotic_disp_x = np.abs(np.diff(mitotic_disp_x))

        non_mitotic_disp_z = np.abs(np.diff(non_mitotic_disp_z))
        non_mitotic_disp_y = np.abs(np.diff(non_mitotic_disp_y))
        non_mitotic_disp_x = np.abs(np.diff(non_mitotic_disp_x))

        all_disp_z = np.abs(np.diff(all_disp_z))
        all_disp_y = np.abs(np.diff(all_disp_y))
        all_disp_x = np.abs(np.diff(all_disp_x))

        goblet_disp_z = np.abs(np.diff(goblet_disp_z))
        goblet_disp_y = np.abs(np.diff(goblet_disp_y))
        goblet_disp_x = np.abs(np.diff(goblet_disp_x))

        basal_disp_z = np.abs(np.diff(basal_disp_z))
        basal_disp_y = np.abs(np.diff(basal_disp_y))
        basal_disp_x = np.abs(np.diff(basal_disp_x))

        radial_disp_z = np.abs(np.diff(radial_disp_z))
        radial_disp_y = np.abs(np.diff(radial_disp_y))
        radial_disp_x = np.abs(np.diff(radial_disp_x))

        self.time.append(i * self.tcalibration)

        self.mitotic_mean_disp_z.append(np.mean(mitotic_disp_z))
        self.mitotic_var_disp_z.append(np.std(mitotic_disp_z))

        self.mitotic_mean_disp_y.append(np.mean(mitotic_disp_y))
        self.mitotic_var_disp_y.append(np.std(mitotic_disp_y))

        self.mitotic_mean_disp_x.append(np.mean(mitotic_disp_x))
        self.mitotic_var_disp_x.append(np.std(mitotic_disp_x))

        filtered_values = [val for val in mitotic_radius if val is not None]
        self.mitotic_mean_radius.append(np.mean(filtered_values))

        self.mitotic_var_radius.append(np.std(filtered_values))

        self.mitotic_mean_speed.append(np.mean(mitotic_speed))
        self.mitotic_var_speed.append(np.std(mitotic_speed))

        self.mitotic_mean_acc.append(np.mean(mitotic_acc))
        self.mitotic_var_acc.append(np.std(mitotic_acc))

        self.mitotic_mean_directional_change_z.append(
            np.mean(mitotic_directional_change_z)
        )
        self.mitotic_var_directional_change_z.append(
            np.std(mitotic_directional_change_z)
        )

        self.mitotic_mean_directional_change_y.append(
            np.mean(mitotic_directional_change_y)
        )
        self.mitotic_var_directional_change_y.append(
            np.std(mitotic_directional_change_y)
        )

        self.mitotic_mean_directional_change_x.append(
            np.mean(mitotic_directional_change_x)
        )
        self.mitotic_var_directional_change_x.append(
            np.std(mitotic_directional_change_x)
        )

        self.mitotic_mean_distance_cell_mask.append(np.mean(mitotic_distance_cell_mask))
        self.mitotic_var_distance_cell_mask.append(np.std(mitotic_distance_cell_mask))

        self.mitotic_mean_local_cell_density.append(np.mean(mitotic_local_cell_density))
        self.mitotic_var_local_cell_density.append(np.std(mitotic_local_cell_density))

        self.non_mitotic_mean_disp_z.append(np.mean(non_mitotic_disp_z))
        self.non_mitotic_var_disp_z.append(np.std(non_mitotic_disp_z))

        self.non_mitotic_mean_disp_y.append(np.mean(non_mitotic_disp_y))
        self.non_mitotic_var_disp_y.append(np.std(non_mitotic_disp_y))

        self.non_mitotic_mean_disp_x.append(np.mean(non_mitotic_disp_x))
        self.non_mitotic_var_disp_x.append(np.std(non_mitotic_disp_x))

        filtered_values = [val for val in non_mitotic_radius if val is not None]
        self.non_mitotic_mean_radius.append(np.mean(filtered_values))

        self.non_mitotic_var_radius.append(np.std(filtered_values))

        self.non_mitotic_mean_speed.append(np.mean(non_mitotic_speed))
        self.non_mitotic_var_speed.append(np.std(non_mitotic_speed))

        self.non_mitotic_mean_acc.append(np.mean(non_mitotic_acc))
        self.non_mitotic_var_acc.append(np.std(non_mitotic_acc))

        self.non_mitotic_mean_directional_change_z.append(
            np.mean(non_mitotic_directional_change_z)
        )
        self.non_mitotic_var_directional_change_z.append(
            np.std(non_mitotic_directional_change_z)
        )

        self.non_mitotic_mean_directional_change_y.append(
            np.mean(non_mitotic_directional_change_y)
        )
        self.non_mitotic_var_directional_change_y.append(
            np.std(non_mitotic_directional_change_y)
        )

        self.non_mitotic_mean_directional_change_x.append(
            np.mean(non_mitotic_directional_change_x)
        )
        self.non_mitotic_var_directional_change_x.append(
            np.std(non_mitotic_directional_change_x)
        )

        self.non_mitotic_mean_distance_cell_mask.append(
            np.mean(non_mitotic_distance_cell_mask)
        )
        self.non_mitotic_var_distance_cell_mask.append(
            np.std(non_mitotic_distance_cell_mask)
        )

        self.non_mitotic_mean_local_cell_density.append(
            np.mean(non_mitotic_local_cell_density)
        )
        self.non_mitotic_var_local_cell_density.append(
            np.std(non_mitotic_local_cell_density)
        )

        self.goblet_mean_disp_z.append(np.mean(goblet_disp_z))
        self.goblet_var_disp_z.append(np.std(goblet_disp_z))

        self.goblet_mean_disp_y.append(np.mean(goblet_disp_y))
        self.goblet_var_disp_y.append(np.std(goblet_disp_y))

        self.goblet_mean_disp_x.append(np.mean(goblet_disp_x))
        self.goblet_var_disp_x.append(np.std(goblet_disp_x))

        filtered_values = [val for val in goblet_radius if val is not None]
        self.goblet_mean_radius.append(np.mean(filtered_values))

        self.goblet_var_radius.append(np.std(filtered_values))

        self.goblet_mean_speed.append(np.mean(goblet_speed))
        self.goblet_var_speed.append(np.std(goblet_speed))

        self.goblet_mean_acc.append(np.mean(goblet_acc))
        self.goblet_var_acc.append(np.std(goblet_acc))

        self.goblet_mean_directional_change_z.append(
            np.mean(goblet_directional_change_z)
        )
        self.goblet_var_directional_change_z.append(np.std(goblet_directional_change_z))

        self.goblet_mean_directional_change_y.append(
            np.mean(goblet_directional_change_y)
        )
        self.goblet_var_directional_change_y.append(np.std(goblet_directional_change_y))

        self.goblet_mean_directional_change_x.append(
            np.mean(goblet_directional_change_x)
        )
        self.goblet_var_directional_change_x.append(np.std(goblet_directional_change_x))

        self.goblet_mean_distance_cell_mask.append(np.mean(goblet_distance_cell_mask))
        self.goblet_var_distance_cell_mask.append(np.std(goblet_distance_cell_mask))

        self.goblet_mean_local_cell_density.append(np.mean(goblet_local_cell_density))
        self.goblet_var_local_cell_density.append(np.std(goblet_local_cell_density))

        self.basal_mean_disp_z.append(np.mean(basal_disp_z))
        self.basal_var_disp_z.append(np.std(basal_disp_z))

        self.basal_mean_disp_y.append(np.mean(basal_disp_y))
        self.basal_var_disp_y.append(np.std(basal_disp_y))

        self.basal_mean_disp_x.append(np.mean(basal_disp_x))
        self.basal_var_disp_x.append(np.std(basal_disp_x))

        filtered_values = [val for val in basal_radius if val is not None]
        self.basal_mean_radius.append(np.mean(filtered_values))

        self.basal_var_radius.append(np.std(filtered_values))

        self.basal_mean_speed.append(np.mean(basal_speed))
        self.basal_var_speed.append(np.std(basal_speed))

        self.basal_mean_acc.append(np.mean(basal_acc))
        self.basal_var_acc.append(np.std(basal_acc))

        self.basal_mean_directional_change_z.append(np.mean(basal_directional_change_z))
        self.basal_var_directional_change_z.append(np.std(basal_directional_change_z))

        self.basal_mean_directional_change_y.append(np.mean(basal_directional_change_y))
        self.basal_var_directional_change_y.append(np.std(basal_directional_change_y))

        self.basal_mean_directional_change_x.append(np.mean(basal_directional_change_x))
        self.basal_var_directional_change_x.append(np.std(basal_directional_change_x))

        self.basal_mean_distance_cell_mask.append(np.mean(basal_distance_cell_mask))
        self.basal_var_distance_cell_mask.append(np.std(basal_distance_cell_mask))

        self.basal_mean_local_cell_density.append(np.mean(basal_local_cell_density))
        self.basal_var_local_cell_density.append(np.std(basal_local_cell_density))

        self.radial_mean_disp_z.append(np.mean(radial_disp_z))
        self.radial_var_disp_z.append(np.std(radial_disp_z))

        self.radial_mean_disp_y.append(np.mean(radial_disp_y))
        self.radial_var_disp_y.append(np.std(radial_disp_y))

        self.radial_mean_disp_x.append(np.mean(radial_disp_x))
        self.radial_var_disp_x.append(np.std(radial_disp_x))

        filtered_values = [val for val in radial_radius if val is not None]
        self.radial_mean_radius.append(np.mean(filtered_values))

        self.radial_var_radius.append(np.std(filtered_values))

        self.radial_mean_speed.append(np.mean(radial_speed))
        self.radial_var_speed.append(np.std(radial_speed))

        self.radial_mean_acc.append(np.mean(radial_acc))
        self.radial_var_acc.append(np.std(radial_acc))

        self.radial_mean_directional_change_z.append(
            np.mean(radial_directional_change_z)
        )
        self.radial_var_directional_change_z.append(np.std(radial_directional_change_z))

        self.radial_mean_directional_change_y.append(
            np.mean(radial_directional_change_y)
        )
        self.radial_var_directional_change_y.append(np.std(radial_directional_change_y))

        self.radial_mean_directional_change_x.append(
            np.mean(radial_directional_change_x)
        )
        self.radial_var_directional_change_x.append(np.std(radial_directional_change_x))

        self.radial_mean_distance_cell_mask.append(np.mean(radial_distance_cell_mask))
        self.radial_var_distance_cell_mask.append(np.std(radial_distance_cell_mask))

        self.radial_mean_local_cell_density.append(np.mean(radial_local_cell_density))
        self.radial_var_local_cell_density.append(np.std(radial_local_cell_density))

        self.all_mean_disp_z.append(np.mean(all_disp_z))
        self.all_var_disp_z.append(np.std(all_disp_z))

        self.all_mean_disp_y.append(np.mean(all_disp_y))
        self.all_var_disp_y.append(np.std(all_disp_y))

        self.all_mean_disp_x.append(np.mean(all_disp_x))
        self.all_var_disp_x.append(np.std(all_disp_x))

        filtered_values = [val for val in all_radius if val is not None]
        self.all_mean_radius.append(np.mean(filtered_values))

        self.all_var_radius.append(np.std(filtered_values))

        self.all_mean_speed.append(np.mean(all_speed))
        self.all_var_speed.append(np.std(all_speed))

        self.all_mean_acc.append(np.mean(all_acc))
        self.all_var_acc.append(np.std(all_acc))

        self.all_mean_directional_change_z.append(np.mean(all_directional_change_z))
        self.all_var_directional_change_z.append(np.std(all_directional_change_z))

        self.all_mean_directional_change_y.append(np.mean(all_directional_change_y))
        self.all_var_directional_change_y.append(np.std(all_directional_change_y))

        self.all_mean_directional_change_x.append(np.mean(all_directional_change_x))
        self.all_var_directional_change_x.append(np.std(all_directional_change_x))

        self.all_mean_distance_cell_mask.append(np.mean(all_distance_cell_mask))
        self.all_var_distance_cell_mask.append(np.std(all_distance_cell_mask))

        self.all_mean_local_cell_density.append(np.mean(all_local_cell_density))
        self.all_var_local_cell_density.append(np.std(all_local_cell_density))


def get_largest_size(timed_cell_size):
    largest_size = max(timed_cell_size.values())
    return largest_size


def compute_cell_size(seg_image, top_n=10):
    ndim = len(seg_image.shape)
    timed_cell_size = {}

    if ndim == 2:
        props = measure.regionprops(seg_image)
        largest_size = []
        for prop in props:
            try:
                largest_size.append(prop.feret_diameter_max)
            except Exception:
                pass

        if largest_size:
            max_size = max(largest_size)
            print(f"The maximum Feret diameter is: {max_size}")
        else:
            print("No valid Feret diameter values were found.")

        timed_cell_size[str(0)] = float(max_size)

    if ndim in (3, 4):
        for i in tqdm(range(0, seg_image.shape[0], int(seg_image.shape[0] / 2))):
            labeled_image = measure.label(seg_image[i, :])

            props = measure.regionprops(labeled_image)

            sorted_props = sorted(props, key=lambda x: x.area, reverse=True)[:top_n]

            top_labels = [prop.label for prop in sorted_props]
            for prop in props:
                if prop.label not in top_labels:
                    labeled_image[labeled_image == prop.label] = 0

            props_filtered = measure.regionprops(labeled_image)

            largest_size = []
            for prop in props_filtered:
                try:
                    largest_size.append(prop.feret_diameter_max)
                except Exception:
                    pass

            if largest_size:
                max_size = max(largest_size)
                print(f"The maximum Feret diameter is: {max_size}")
            else:
                print("No valid Feret diameter values were found.")

            timed_cell_size[str(i)] = float(max_size)

    return timed_cell_size


def boundary_points(mask, xcalibration, ycalibration, zcalibration):

    ndim = len(mask.shape)
    timed_mask = {}
    mask = mask > 0
    mask = mask.astype("uint8")
    # YX shaped object
    if ndim == 2:

        boundary = find_boundaries(mask)
        regioncentroid = (0,) + compute_centroid(boundary)
        indices = np.where(boundary > 0)
        real_indices = np.transpose(np.asarray(indices, dtype=np.float32)).copy()

        for j in range(0, len(real_indices)):

            real_indices[j][0] = real_indices[j][0] * ycalibration
            real_indices[j][1] = real_indices[j][1] * xcalibration

        tree = spatial.cKDTree(real_indices)
        # This object contains list of all the points for all the labels in the Mask image with the label id and  of each label
        timed_mask[str(0)] = [tree, indices, regioncentroid]

    # TYX shaped object
    if ndim == 3:

        for i in tqdm(range(0, mask.shape[0])):

            boundary = find_boundaries(mask[i, :])
            regioncentroid = (0,) + compute_centroid(boundary)
            indices = np.where(boundary > 0)
            real_indices = np.transpose(np.asarray(indices, dtype=np.float32)).copy()

            for j in range(0, len(real_indices)):

                real_indices[j][0] = real_indices[j][0] * ycalibration
                real_indices[j][1] = real_indices[j][1] * xcalibration

            tree = spatial.cKDTree(real_indices)

            timed_mask[str(i)] = [tree, indices, regioncentroid]

    # TZYX shaped object
    if ndim == 4:
        boundary = np.zeros(
            [mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]], dtype=np.uint8
        )
        for i in range(0, mask.shape[0]):

            boundary[i, :] = find_boundaries(mask[i, :])
            regioncentroid = compute_centroid(boundary[i, :])
            indices = np.where(boundary[i, :] > 0)
            real_indices = np.transpose(np.asarray(indices, dtype=np.float32)).copy()

            for j in range(0, len(real_indices)):

                real_indices[j][0] = real_indices[j][0] * zcalibration
                real_indices[j][1] = real_indices[j][1] * ycalibration
                real_indices[j][2] = real_indices[j][2] * xcalibration

            tree = spatial.cKDTree(real_indices)
            timed_mask[str(i)] = [tree, indices, regioncentroid]

    return timed_mask, boundary


def compute_centroid(binary_image):
    # Ensure binary image is a NumPy array
    binary_image = np.array(binary_image)

    white_pixels = np.where(binary_image == 1)
    num_pixels = len(white_pixels[0])

    # Compute the centroid of the white pixels in the boundary image
    centroid = np.zeros(binary_image.ndim)
    for dim in range(binary_image.ndim):
        centroid[dim] = white_pixels[dim].sum() / num_pixels

    return centroid


def get_csv_data(csv):

    dataset = pd.read_csv(
        csv, delimiter=",", encoding="unicode_escape", low_memory=False
    )[3:]
    dataset_index = dataset.index
    return dataset, dataset_index


def get_spot_dataset(
    spot_dataset,
    track_analysis_spot_keys,
    xcalibration,
    ycalibration,
    zcalibration,
    AttributeBoxname,
    detectionchannel,
):
    AllValues = {}
    posix = track_analysis_spot_keys["posix"]
    posiy = track_analysis_spot_keys["posiy"]
    posiz = track_analysis_spot_keys["posiz"]
    frame = track_analysis_spot_keys["frame"]

    LocationX = (spot_dataset[posix].astype("float") / xcalibration).astype("int")
    LocationY = (spot_dataset[posiy].astype("float") / ycalibration).astype("int")
    LocationZ = (spot_dataset[posiz].astype("float") / zcalibration).astype("int")
    LocationT = (spot_dataset[frame].astype("float")).astype("int")

    ignore_values = [
        track_analysis_spot_keys["mean_intensity"],
        track_analysis_spot_keys["total_intensity"],
    ]
    for (k, v) in track_analysis_spot_keys.items():

        if detectionchannel == 1:
            if k == "mean_intensity_ch2":
                value = track_analysis_spot_keys["mean_intensity"]
                AllValues[value] = spot_dataset[v].astype("float")
            if k == "total_intensity_ch2":
                value = track_analysis_spot_keys["total_intensity"]
                AllValues[value] = spot_dataset[v].astype("float")

        if v not in ignore_values:

            AllValues[v] = spot_dataset[v].astype("float")

    AllValues[posix] = round(LocationX, 3)
    AllValues[posiy] = round(LocationY, 3)
    AllValues[posiz] = round(LocationZ, 3)
    AllValues[frame] = round(LocationT, 3)
    Attributeids = []
    Attributeids.append(AttributeBoxname)
    for attributename in AllValues.keys():
        Attributeids.append(attributename)

    return Attributeids, AllValues


def get_track_dataset(
    track_dataset,
    track_analysis_spot_keys,
    track_analysis_track_keys,
    TrackAttributeBoxname,
):

    AllTrackValues = {}
    track_id = track_analysis_spot_keys["track_id"]
    Tid = track_dataset[track_id].astype("float")

    AllTrackValues[track_id] = Tid

    for (k, v) in track_analysis_track_keys.items():

        x = track_dataset[v].astype("float")
        minval = min(x)
        maxval = max(x)

        if minval > 0 and maxval <= 1:

            x = x + 1

        AllTrackValues[k] = round(x, 3)

    TrackAttributeids = []
    TrackAttributeids.append(TrackAttributeBoxname)
    for attributename in track_analysis_track_keys.keys():
        TrackAttributeids.append(attributename)

    return TrackAttributeids, AllTrackValues


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


def get_edges_dataset(
    edges_dataset,
    edges_dataset_index,
    track_analysis_spot_keys,
    track_analysis_edges_keys,
):

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


def scale_value(x, scale=255 * 255):

    return x * scale


def prob_sigmoid(x):
    return 1 - math.exp(-x)


def cell_angular_change_z(vec_cell):

    vec = np.asarray(vec_cell)
    vec = vec / np.linalg.norm(vec)
    num_dims = len(vec)
    unit_vector = np.zeros(num_dims)
    unit_vector[-1] = 1
    unit_vector = unit_vector / np.linalg.norm(unit_vector)
    theta = np.arccos(np.clip(np.dot(vec, unit_vector), -1.0, 1.0))
    angle = np.rad2deg(theta)

    return angle


def cell_angular_change_y(vec_cell):

    vec = np.asarray(vec_cell)
    vec = vec / np.linalg.norm(vec)
    num_dims = len(vec)
    unit_vector = np.zeros(num_dims)
    unit_vector[-2] = 1
    unit_vector = unit_vector / np.linalg.norm(unit_vector)
    theta = np.arccos(np.clip(np.dot(vec, unit_vector), -1.0, 1.0))
    angle = np.rad2deg(theta)

    return angle


def cell_angular_change_x(vec_cell):

    vec = np.asarray(vec_cell)
    vec = vec / np.linalg.norm(vec)
    num_dims = len(vec)
    unit_vector = np.zeros(num_dims)
    unit_vector[0] = 1
    unit_vector = unit_vector / np.linalg.norm(unit_vector)
    theta = np.arccos(np.clip(np.dot(vec, unit_vector), -1.0, 1.0))
    angle = np.rad2deg(theta)

    return angle


def check_and_update_mask(mask, image):
    if len(mask.shape) < len(image.shape):
        update_mask = np.zeros_like(image, dtype="uint8")
        for i in range(image.shape[0]):
            labeled_mask, num_features = measure.label(
                mask[i], background=0, return_num=True
            )
            if num_features > 0:
                props = measure.regionprops(labeled_mask)
                largest_region = max(props, key=lambda prop: prop.area)
                largest_label = largest_region.label
                update_mask[i] = (labeled_mask == largest_label).astype("uint8")
    else:
        labeled_mask, num_features = measure.label(mask, background=0, return_num=True)
        if num_features > 0:
            props = measure.regionprops(labeled_mask)
            largest_region = max(props, key=lambda prop: prop.area)
            largest_label = largest_region.label
            update_mask = (labeled_mask == largest_label).astype("uint8")
        else:
            update_mask = mask

    return update_mask


def get_feature_dict(unique_tracks_properties):

    features = {
        "time": np.asarray(unique_tracks_properties, dtype="float16")[:, 0],
        "generation": np.asarray(unique_tracks_properties, dtype="float16")[:, 2],
        "radius": np.asarray(unique_tracks_properties, dtype="float16")[:, 3],
        "eccentricity_comp_first": np.asarray(
            unique_tracks_properties, dtype="float16"
        )[:, 4],
        "eccentricity_comp_second": np.asarray(
            unique_tracks_properties, dtype="float16"
        )[:, 5],
        "eccentricity_comp_third": np.asarray(
            unique_tracks_properties, dtype="float16"
        )[:, 6],
        "surface_area": np.asarray(unique_tracks_properties, dtype="float16")[:, 7],
        "total_intensity": np.asarray(unique_tracks_properties, dtype="float16")[:, 8],
        "speed": np.asarray(unique_tracks_properties, dtype="float16")[:, 9],
        "motion_angle_z": np.asarray(unique_tracks_properties, dtype="float16")[:, 10],
        "motion_angle_y": np.asarray(unique_tracks_properties, dtype="float16")[:, 11],
        "motion_angle_x": np.asarray(unique_tracks_properties, dtype="float16")[:, 12],
        "acceleration": np.asarray(unique_tracks_properties, dtype="float16")[:, 13],
        "distance_cell_mask": np.asarray(unique_tracks_properties, dtype="float16")[
            :, 14
        ],
        "local_cell_density": np.asarray(unique_tracks_properties, dtype="float16")[
            :, 15
        ],
        "radial_angle_z": np.asarray(unique_tracks_properties, dtype="float16")[:, 16],
        "radial_angle_y": np.asarray(unique_tracks_properties, dtype="float16")[:, 17],
        "radial_angle_x": np.asarray(unique_tracks_properties, dtype="float16")[:, 18],
        "cell_axis_z": np.asarray(unique_tracks_properties, dtype="float16")[:, 19],
        "cell_axis_y": np.asarray(unique_tracks_properties, dtype="float16")[:, 20],
        "cell_axis_x": np.asarray(unique_tracks_properties, dtype="float16")[:, 21],
        "track_displacement": np.asarray(unique_tracks_properties, dtype="float16")[
            :, 22
        ],
        "total_track_distance": np.asarray(unique_tracks_properties, dtype="float16")[
            :, 23
        ],
        "max_track_distance": np.asarray(unique_tracks_properties, dtype="float16")[
            :, 24
        ],
        "track_duration": np.asarray(unique_tracks_properties, dtype="float16")[:, 25],
        "msd": np.asarray(unique_tracks_properties, dtype="float16")[:, 26],
    }

    return features


def set_scale(dimensions, x_calibration, y_calibration, z_calibration):

    scale = [x_calibration, y_calibration, z_calibration]
    arranged_scale = [scale[dim] for dim in dimensions]
    return tuple(arranged_scale)


def transfer_fate_location(membranesegimage, csv_file, save_file, space_veto=10):
    """
    Transfer fate location based on membrane segmentation image.

    Parameters:
    - membranesegimage: ndarray
        Membrane segmentation image.
    - csv_file: str
        Path to the CSV file containing coordinates.
    - save_file: str
        Path to save the output CSV file.
    - space_veto: int, optional
        Space veto threshold.

    Returns:
    None
    """
    dataframe = pd.read_csv(csv_file)
    writer = csv.writer(open(save_file, "w", newline=""))
    writer.writerow(["t", "z", "y", "x"])
    dict_membrane = {}
    if isinstance(membranesegimage, str):
        membranesegimage = imread(membranesegimage)

    for i in tqdm(range(membranesegimage.shape[0])):
        currentimage = membranesegimage[i, :, :, :]
        properties = measure.regionprops(currentimage)
        membrane_coordinates = [prop.centroid for prop in properties]
        dict_membrane[i] = membrane_coordinates

    for index, row in dataframe.iterrows():

        t = int(round(row["t"]))
        z = round(row["z"])
        y = round(row["y"])
        x = round(row["x"])

        membrane_coordinates = dict_membrane[t]
        if len(membrane_coordinates) > 0:
            tree = spatial.cKDTree(membrane_coordinates)
            index = (z, y, x)
            distance, nearest_location = tree.query(index)
            if distance <= space_veto:
                z = int(membrane_coordinates[nearest_location][0])
                y = membrane_coordinates[nearest_location][1]
                x = membrane_coordinates[nearest_location][2]
                writer.writerow([t, z, y, x])
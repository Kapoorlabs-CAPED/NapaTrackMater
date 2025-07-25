from .pretrained import register_model, register_aliases, clear_models_and_aliases
from .clustering import Clustering
from .Trackmate import TrackMate, get_feature_dict, transfer_fate_location
from .Trackcomparator import TrackComparator
from .ergodicity import Ergodicity
from .Trackvector import (
    TrackVector,
    convert_tracks_to_arrays,
    unsupervised_clustering,
    simple_unsupervised_clustering,
    create_global_gt_dataframe,
    create_gt_analysis_vectors_dict,
    train_mitosis_neural_net,
    train_gbr_neural_net,
    plot_metrics_from_npz,
    create_analysis_tracklets,
    create_dividing_prediction_tracklets,
    convert_tracks_to_simple_arrays,
    local_track_covaraince,
    cell_fate_recipe,
    core_clustering,
    pseudo_core_clustering,
    convert_pseudo_tracks_to_simple_arrays,
    create_analysis_cell_type_tracklets,
    update_cluster_plot,
    update_distance_cluster_plot,
    update_eucledian_distance_cluster_plot,
    plot_at_mitosis_time,
    plot_histograms_for_groups,
    create_microdomain_movie,
    create_cluster_plot,
    create_microdomain_video,
    create_movie,
    create_video,
    plot_histograms_for_cell_type_groups,
    inception_model_prediction,
    save_cell_type_predictions,
    filter_and_get_tracklets,
  
    normalize_image_in_chunks,
    inception_dual_model_prediction
)

from .drift import (
    affine_transform,  apply_alpha_drift, apply_xy_drift, apply_z_drift, crop_data, get_rotation, get_xy_drift, get_z_drift,
    apply_alpha_drift_numpy, apply_xy_drift_numpy, apply_z_drift_numpy, crop_data_numpy, get_rotation_numpy, get_xy_drift_numpy, get_z_drift_numpy
)
from .homology import vietoris_rips_at_t, diagrams_over_time, save_barcodes_and_stats

from .CloudAutoEncoder import CloudAutoEncoder
import json
from csbdeep.utils.tf import keras_import
from tifffile import imread
import os

get_file = keras_import("utils", "get_file")


def abspath(path):

    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def test_tracks_xenopus():
    url = "https://zenodo.org/record/8417322/files/example_tracks_napari_trackmate.zip"
    hash = "589c7b1834b995adb2ae183604961f48"
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    abspath(
        get_file(
            fname="example_tracks_napari_trackmate.zip",
            origin=url,
            file_hash=hash,
            extract=True,
            archive_format="zip",
            cache_subdir=parent_directory,
        )
    )
    extracted_folder = os.path.join(parent_directory, "nuclei_membrane_tracking")
    image = imread(os.path.join(extracted_folder, "nuclei_timelapse_test_dataset.tif"))
    return image


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)


__all__ = (
    "CloudAutoEncoder",
    "Clustering",
    "TrackMate",
    "get_feature_dict",
    "create_gt_analysis_vectors_dict",
    "create_global_gt_dataframe",
    "supervised_clustering",
    "TrackVector",
    "predict_supervised_clustering",
    "convert_tracks_to_arrays",
    "convert_tracks_to_simple_arrays",
    "unsupervised_clustering",
    "simple_unsupervised_clustering",
    "load_json",
    "train_mitosis_neural_net",
    "train_gbr_neural_net",
    "predict_with_model",
    "plot_metrics_from_npz",
    "load_prediction_data",
    "create_embeddings_with_gt",
    "create_analysis_tracklets",
    "local_track_covaraince",
    "create_dividing_prediction_tracklets",
    "cell_fate_recipe",
    "core_clustering",
    "pseudo_core_clustering",
    "convert_pseudo_tracks_to_simple_arrays",
    "create_analysis_cell_type_tracklets",
    "update_cluster_plot",
    "update_distance_cluster_plot",
    "update_eucledian_distance_cluster_plot",
    "plot_at_mitosis_time",
    "plot_histograms_for_groups",
    "create_microdomain_movie",
    "create_cluster_plot",
    "create_microdomain_video",
    "create_movie",
    "create_video",
    "plot_histograms_for_cell_type_groups",
    "inception_model_prediction",
    "save_cell_type_predictions",
    "transfer_fate_location",
    "filter_and_get_tracklets",
    "create_h5",
    "normalize_image_in_chunks",
    "inception_dual_model_prediction",
    "affine_transform",
    "apply_alpha_drift", 
    "apply_xy_drift", 
    "apply_z_drift", 
    "crop_data", 
    "get_rotation", 
    "get_xy_drift", 
    "get_z_drift",
    "apply_alpha_drift_numpy", 
    "apply_xy_drift_numpy", 
    "apply_z_drift_numpy", 
    "crop_data_numpy", 
    "get_rotation_numpy", 
    "get_xy_drift_numpy", 
    "get_z_drift_numpy",
    "TrackComparator",
    "vietoris_rips_at_t",  "diagrams_over_time", "save_barcodes_and_stats", "Ergodicity"


)

clear_models_and_aliases(CloudAutoEncoder)

register_model(
    CloudAutoEncoder,
    "xenopus_nuclei_autoencoder",
    "https://zenodo.org/record/8025253/files/xenopus_nuclei_autoencoder.zip",
    "9527d5767d92b04689cd53cead3a2496",
)
register_model(
    CloudAutoEncoder,
    "xenopus_membrane_autoencoder",
    "https://zenodo.org/record/8025269/files/xenopus_membrane_autoencoder.zip",
    "b8763726e1a9e15202960a6f384093d6",
)


register_aliases(
    CloudAutoEncoder, "xenopus_nuclei_autoencoder", "xenopus_nuclei_autoencoder"
)
register_aliases(
    CloudAutoEncoder, "xenopus_membrane_autoencoder", "xenopus_membrane_autoencoder"
)


del register_model, register_aliases, clear_models_and_aliases

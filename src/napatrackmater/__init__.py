from .pretrained import register_model, register_aliases, clear_models_and_aliases
from .clustering import Clustering
from .Trackmate import TrackMate, get_feature_dict
from .Trackvector import (
    TrackVector,
    create_analysis_vectors_dict,
    convert_tracks_to_arrays,
    perform_cosine_similarity,
    perform_pca,
    perform_kmeans,
)
from .DeepEmbeddedClustering import DeepEmbeddedClustering
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
    "TrackVector",
    "create_analysis_vectors_dict",
    "convert_tracks_to_arrays",
    "perform_cosine_similarity",
    "perform_pca",
    "perform_kmeans",
    "DeepEmbeddedClustering",
    "load_json",
)

clear_models_and_aliases(DeepEmbeddedClustering, CloudAutoEncoder)

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

# register_model(DeepEmbeddedClustering, 'cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3', 'https://zenodo.org/record/7677125/files/cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3.zip', '26608aba6282788c2d7bf2e7bbf38711')
# register_model(DeepEmbeddedClustering, 'Membrane_3D_cluster', '.zip', 'hash')


register_aliases(
    CloudAutoEncoder, "xenopus_nuclei_autoencoder", "xenopus_nuclei_autoencoder"
)
register_aliases(
    CloudAutoEncoder, "xenopus_membrane_autoencoder", "xenopus_membrane_autoencoder"
)

# register_aliases(DeepEmbeddedClustering, 'cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3', 'cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3')
# register_aliases(DeepEmbeddedClustering, 'Membrane_3D_cluster', 'Membrane_3D_cluster')

del register_model, register_aliases, clear_models_and_aliases

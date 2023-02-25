
from .Trackmate import *
from .version import __version__
from .pretrained import register_model, register_aliases, clear_models_and_aliases, get_registered_models, get_model_folder
from .clustering import Clustering
from cellshape_cluster import DeepEmbeddedClustering
from cellshape_cloud import CloudAutoEncoder
import json




def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)

__all__ = (
    "DeepEmbeddedClustering",
    "CloudAutoEncoder",
)

clear_models_and_aliases(DeepEmbeddedClustering, CloudAutoEncoder)

register_model(CloudAutoEncoder, 'Nuclei_3D', 'https://zenodo.org/record/7677125/files/xenopus_nuclei_dgcnn_foldingnet_knn8.zip', 'fb476f050aacf81b61a3233ae69c6ad4' )
register_model(CloudAutoEncoder, 'Membrane_3D', '.zip', 'hash')

register_model(DeepEmbeddedClustering, 'Nuclei_3D_cluster_3_class', 'https://zenodo.org/record/7677125/files/cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3.zip', '26608aba6282788c2d7bf2e7bbf38711')
register_model(DeepEmbeddedClustering, 'Membrane_3D_cluster', '.zip', 'hash')


register_aliases(CloudAutoEncoder, 'Nuclei_3D', 'Nuclei_3D')
register_aliases(CloudAutoEncoder, 'Membrane_3D', 'Membrane_3D')

register_aliases(DeepEmbeddedClustering, 'Nuclei_3D_cluster_3_class', 'Nuclei_3D_cluster_3_class')
register_aliases(DeepEmbeddedClustering, 'Membrane_3D_cluster', 'Membrane_3D_cluster')

del register_model, register_aliases, clear_models_and_aliases


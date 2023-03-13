
from .Trackmate import *
from .Trackvector import *
from .version import __version__
from .pretrained import register_model, register_aliases, clear_models_and_aliases
from .clustering import Clustering
from .DeepEmbeddedClustering import DeepEmbeddedClustering
from .CloudAutoEncoder import CloudAutoEncoder
import json




def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)

__all__ = (
    "DeepEmbeddedClustering",
    "CloudAutoEncoder",
    "Clustering",
)

clear_models_and_aliases(DeepEmbeddedClustering, CloudAutoEncoder)

register_model(CloudAutoEncoder, 'xenopus_nuclei_dgcnn_foldingnet_knn8', 'https://zenodo.org/record/7677125/files/xenopus_nuclei_dgcnn_foldingnet_knn8.zip', 'fb476f050aacf81b61a3233ae69c6ad4' )
register_model(CloudAutoEncoder, 'Membrane_3D', '.zip', 'hash')

register_model(DeepEmbeddedClustering, 'cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3', 'https://zenodo.org/record/7677125/files/cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3.zip', '26608aba6282788c2d7bf2e7bbf38711')
register_model(DeepEmbeddedClustering, 'Membrane_3D_cluster', '.zip', 'hash')


register_aliases(CloudAutoEncoder, 'xenopus_nuclei_dgcnn_foldingnet_knn8', 'xenopus_nuclei_dgcnn_foldingnet_knn8')
register_aliases(CloudAutoEncoder, 'Membrane_3D', 'Membrane_3D')

register_aliases(DeepEmbeddedClustering, 'cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3', 'cluster_xenopus_nuclei_dgcnn_foldingnet_knn8_class3')
register_aliases(DeepEmbeddedClustering, 'Membrane_3D_cluster', 'Membrane_3D_cluster')

del register_model, register_aliases, clear_models_and_aliases


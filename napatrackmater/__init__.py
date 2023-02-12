from .bTrackmate import *
from .Trackmate import *
from .napari_animation import *
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

register_model(CloudAutoEncoder, 'Nuclei_3D', '.zip', 'hash')
register_model(CloudAutoEncoder, 'Membrane_3D', '.zip', 'hash')

register_model(DeepEmbeddedClustering, 'Nuclei_3D_cluster', '.zip', 'hash')
register_model(DeepEmbeddedClustering, 'Membrane_3D_cluster', '.zip', 'hash')


register_aliases(CloudAutoEncoder, 'Nuclei_3D', 'Nuclei_3D')
register_aliases(CloudAutoEncoder, 'Membrane_3D', 'Membrane_3D')

register_aliases(DeepEmbeddedClustering, 'Nuclei_3D_cluster', 'Nuclei_3D_cluster')
register_aliases(DeepEmbeddedClustering, 'Membrane_3D_cluster', 'Membrane_3D_cluster')

del register_model, register_aliases, clear_models_and_aliases
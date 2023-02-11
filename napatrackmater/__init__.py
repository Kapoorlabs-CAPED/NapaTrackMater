from .bTrackmate import *
from .Trackmate import *
from .napari_animation import *
from .version import __version__
from .pretrained import register_model, register_aliases, clear_models_and_aliases, get_registered_models, get_model_folder, keras_import
from cellshape_cluster import DeepEmbeddedClustering
from cellshape_cloud import CloudAutoEncoder




get_file = keras_import('utils', 'get_file')


__all__ = (
    "DeepEmbeddedClustering",
    "CloudAutoEncoder",
)

clear_models_and_aliases(DeepEmbeddedClustering, CloudAutoEncoder)

register_model(CloudAutoEncoder, 'Nuclei_3D', '.zip', 'hash')
register_model(CloudAutoEncoder, 'Membrane_3D', '.zip', 'hash')

register_model(DeepEmbeddedClustering, 'Nuclei_3D_cluster', '.zip', 'hash')
register_model(DeepEmbeddedClustering, 'Membrane_3D_cluster', '.zip', 'hash')


register_aliases(CloudAutoEncoder, 'Nuclei (3D)', 'Nuclei (3D)')
register_aliases(CloudAutoEncoder, 'Membrane (3D)', 'Membrane (3D)')

register_aliases(DeepEmbeddedClustering, 'Nuclei Cluster (3D)', 'Nuclei Cluster (3D)')
register_aliases(DeepEmbeddedClustering, 'Membrane Cluster (3D)', 'Membrane Cluster (3D)')

del register_model, register_aliases, clear_models_and_aliases
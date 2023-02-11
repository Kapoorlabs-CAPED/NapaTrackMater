from .bTrackmate import *
from .Trackmate import *
from .napari_animation import *
from .version import __version__
from .pretrained import register_model, register_aliases, clear_models_and_aliases
from cellshape_cluster import DeepEmbeddedClustering
from cellshape_cloud import CloudAutoEncoder
from importlib import import_module
import tensorflow

def keras_import(sub=None, *names):
    if sub is None:
        return import_module(tensorflow.keras)
    else:
        mod = import_module('{_KERAS}.{sub}'.format(_KERAS=tensorflow.keras,sub=sub))
        if len(names) == 0:
            return mod
        elif len(names) == 1:
            return getattr(mod, names[0])
        return tuple(getattr(mod, name) for name in names)

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
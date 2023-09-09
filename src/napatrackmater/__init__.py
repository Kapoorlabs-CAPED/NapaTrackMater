from .pretrained import register_model, register_aliases, clear_models_and_aliases
from .clustering import Clustering
from .DeepEmbeddedClustering import DeepEmbeddedClustering
from .CloudAutoEncoder import CloudAutoEncoder
import json


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)


__all__ = (
    # "DeepEmbeddedClustering",
    "CloudAutoEncoder",
    "Clustering",
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

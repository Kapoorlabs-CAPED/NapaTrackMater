from cellshape_cluster import DeepEmbeddedClustering
from .pretrained import get_cluster_instance, get_model_details, get_registered_models
import sys

class DeepEmbeddedClustering(DeepEmbeddedClustering):
    def __init__(slef, autoencoder, num_clusters):
        super().__init__(autoencoder, num_clusters)

    @classmethod   
    def local_from_pretrained(cls,  auto_cls, name_or_alias = None, auto_name_or_alias = None):
           try:
               get_model_details(cls, name_or_alias, verbose=True)
               return get_cluster_instance(cls, name_or_alias, auto_cls, auto_name_or_alias)
           except ValueError:
               if name_or_alias is not None:
                   print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                   sys.stderr.flush()
               get_registered_models(cls, verbose=True)     


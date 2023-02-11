from cellshape_cloud import CloudAutoEncoder
import sys
from .pretrained import get_autoencoder_instance, get_model_details, get_registered_models


class CloudAutoEncoder(CloudAutoEncoder):
    def __init__(
        self,
        num_features,
        k=20,
        encoder_type="dgcnn",
        decoder_type="foldingnet",
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__(
        num_features = num_features,
        k=k,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        shape=shape,
        sphere_path=sphere_path,
        gaussian_path=gaussian_path,
        std=std)

    @classmethod   
    def local_from_pretrained(cls, name_or_alias=None):
           try:
               get_model_details(cls, name_or_alias, verbose=True)
               return get_autoencoder_instance(cls, name_or_alias)
           except ValueError:
               if name_or_alias is not None:
                   print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                   sys.stderr.flush()
               get_registered_models(cls, verbose=True) 
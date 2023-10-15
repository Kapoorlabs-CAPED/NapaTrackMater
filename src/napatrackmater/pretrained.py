"""
Created on Tue Dec 21 16:54:50 2021

@author: stardist devs
"""


from collections import OrderedDict
from warnings import warn
import torch
from pathlib import Path
import tensorflow as tf
import os
import json
from kapoorlabs_lightning.lightning_trainer import AutoLightningModel
from kapoorlabs_lightning.pytorch_losses import ChamferLoss
from kapoorlabs_lightning.optimizers import Adam


def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


_MODELS = {}
_ALIASES = {}


def clear_models_and_aliases(*cls):
    if len(cls) == 0:
        _MODELS.clear()
        _ALIASES.clear()
    else:
        for c in cls:
            if c in _MODELS:
                del _MODELS[c]
            if c in _ALIASES:
                del _ALIASES[c]


def register_model(cls, key, url, hash):
    # key must be a valid file/folder name in the file system
    models = _MODELS.setdefault(cls, OrderedDict())
    key not in models or warn(
        "re-registering model '{}' (was already registered for '{}')".format(
            key, cls.__name__
        )
    )
    models[key] = dict(url=url, hash=hash)


def register_aliases(cls, key, *names):
    # aliases can be arbitrary strings
    if len(names) == 0:
        return
    models = _MODELS.get(cls, {})
    key in models or _raise(
        ValueError(f"model '{key}' is not registered for '{cls.__name__}'")
    )
    aliases = _ALIASES.setdefault(cls, OrderedDict())
    for name in names:
        aliases.get(name, key) == key or warn(
            "alias '{}' was previously registered with model '{}' for '{}'".format(
                name, aliases[name], cls.__name__
            )
        )
        aliases[name] = key


def get_registered_models(cls, return_aliases=True, verbose=False):
    models = _MODELS.get(cls, {})
    aliases = _ALIASES.get(cls, {})
    model_keys = tuple(models.keys())
    model_aliases = {
        key: tuple(name for name in aliases if aliases[name] == key) for key in models
    }
    if verbose:
        # this code is very messy and should be refactored...
        _n = len(models)
        _str_model = "model" if _n == 1 else "models"
        _str_is_are = "is" if _n == 1 else "are"
        _str_colon = ":" if _n > 0 else ""
        print(
            "There {is_are} {n} registered {model_s} for '{clazz}'{c}".format(
                n=_n,
                clazz=cls.__name__,
                is_are=_str_is_are,
                model_s=_str_model,
                c=_str_colon,
            )
        )
        if _n > 0:
            print()
            _maxkeylen = 2 + max(len(key) for key in models)
            print("Name{s}Alias(es)".format(s=" " * (_maxkeylen - 4 + 3)))
            print("────{s}─────────".format(s=" " * (_maxkeylen - 4 + 3)))
            for key in models:
                _aliases = "   "
                _m = len(model_aliases[key])
                if _m > 0:
                    _aliases += "'%s'" % "', '".join(model_aliases[key])
                else:
                    _aliases += "None"
                _key = ("{s:%d}" % _maxkeylen).format(s="'%s'" % key)
                print(f"{_key}{_aliases}")
    return (model_keys, model_aliases) if return_aliases else model_keys


def get_model_details(cls, key_or_alias, verbose=False):
    models = _MODELS.get(cls, {})
    if key_or_alias in models:
        key = key_or_alias
        alias = None
    else:
        aliases = _ALIASES.get(cls, {})
        alias = key_or_alias
        alias in aliases or _raise(
            ValueError(f"'{alias}' is neither a key or alias for '{cls.__name__}'")
        )
        key = aliases[alias]
    if verbose:
        print(
            "Found model '{model}'{alias_str} for '{clazz}'.".format(
                model=key,
                clazz=cls.__name__,
                alias_str=("" if alias is None else " with alias '%s'" % alias),
            )
        )
    return key, alias, models[key]


def get_model_folder(cls, key_or_alias):
    key, alias, m = get_model_details(cls, key_or_alias)
    target = str(Path("models") / cls.__name__ / key)
    path = Path(
        tf.keras.utils.get_file(
            fname=key + ".zip",
            origin=m["url"],
            file_hash=m["hash"],
            cache_subdir=target,
            extract=True,
        )
    )
    assert path.exists() and path.parent.exists()
    return path.parent, key


def load_json(fpath):
    with open(fpath) as f:

        return json.load(f)


def get_autoencoder_instance(cls, key_or_alias):
    path, key = get_model_folder(cls, key_or_alias)
    json_file = os.path.join(
        os.path.join(path.parent.as_posix(), path.name), key + ".json"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if Path(json_file).is_file():

        modelconfig = load_json(json_file)
        scale_z = modelconfig["scale_z"]
        scale_xy = modelconfig["scale_xy"]
        loss = ChamferLoss()
        optimizer = Adam(lr=0.001)
        cloud_autoencoder = cls(
            num_features=modelconfig["num_features"],
            k=modelconfig["k_nearest_neighbours"],
            encoder_type=modelconfig["encoder_type"],
            decoder_type=modelconfig["decoder_type"],
        )
        autoencoder_model = AutoLightningModel.load_from_checkpoint(
            os.path.join(
                os.path.join(path.parent.as_posix(), path.name), key + ".ckpt"
            ),
            map_location=device,
            network=cloud_autoencoder,
            loss_func=loss,
            optim_func=optimizer,
            scale_z=scale_z,
            scale_xy=scale_xy,
        )
        return autoencoder_model
    else:
        print(
            f"Expected a json file of model attributes with name {key}.json in the folder {path.parent}"
        )


def get_cluster_instance(cls, key_or_alias, autoencoder):

    path, key = get_model_folder(cls, key_or_alias)

    checkpoint = torch.load(
        os.path.join(os.path.join(path.parent.as_posix(), path.name), key + ".ckpt")
    )
    num_clusters = checkpoint["model_state_dict"]["clustering_layer.weight"].shape[0]
    print(f"The number of clusters in the loaded model is: {num_clusters}")

    model = cls(autoencoder=autoencoder, num_clusters=num_clusters)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

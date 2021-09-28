import json
import os
from types import SimpleNamespace

def dump_hyperparams(dump_dir,hp_dict):
    with open(os.path.join(dump_dir,'hyperparams.json'),'wt') as fo:
        json.dump(hp_dict,fo,indent=4,sort_keys=True)

def dict_to_hyperparameters(hp_dict):
    return SimpleNamespace(**hp_dict)

def read_hyperparams(read_dir):
    with open(os.path.join(read_dir,'hyperparams.json'),'rt') as fi:
        hp_dict = json.load(fi)
    return dict_to_hyperparameters(hp_dict)

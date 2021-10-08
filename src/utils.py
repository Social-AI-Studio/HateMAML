import json
import os
from types import SimpleNamespace
from src.data.consts import EMBEDDINGS_DIR


def dump_hyperparams(dump_dir, hp_dict):
    with open(os.path.join(dump_dir, "hyperparams.json"), "wt") as fo:
        json.dump(hp_dict, fo, indent=4, sort_keys=True)


def dict_to_hyperparameters(hp_dict):
    return SimpleNamespace(**hp_dict)


def read_hyperparams(read_dir):
    with open(os.path.join(read_dir, "hyperparams.json"), "rt") as fi:
        hp_dict = json.load(fi)
    return dict_to_hyperparameters(hp_dict)


def load_glove_format_embs(fn, pad_token, unk_token):
    fn = os.path.join(EMBEDDINGS_DIR, fn)
    ret_vocab = [pad_token, unk_token]
    ret_embeddings = [
        list(),
        list(),
    ]
    with open(fn, "rt") as fd:
        for line in fd:
            line = line.strip().split(" ")
            ret_vocab.append(line[0])
            ret_embeddings.append([])
            ret_embeddings[-1].append([float(e) for e in line[1:]])

    # embedding for pad_token initialized as the 0 vector.
    ret_embeddings[0] = [0.0 for i in range(len(ret_embeddings[2]))]
    ret_embeddings[1] = [0.0 for i in range(len(ret_embeddings[2]))]

    ret_vocab, ret_embeddings = np.array(ret_vocab), np.array(ret_embeddings)
    # embedding for unk_token initialized as the mean of all embeddings.
    ret_embeddings[1, :] = np.mean(ret_embeddings, axis=0)

    return ret_vocab, ret_embeddings

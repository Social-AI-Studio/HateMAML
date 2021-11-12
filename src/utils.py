import json
import numpy as np
import os
from types import SimpleNamespace
from src.data.consts import EMBEDDINGS_DIR


def dump_hyperparams(dump_dir, hp_dict):
    with open(os.path.join(dump_dir, "hyperparams.json"), "wt") as fo:
        json.dump(hp_dict, fo, indent=4, sort_keys=True)


def dump_vocab(dump_dir, vocab):
    with open(os.path.join(dump_dir, "vocab.json"), "wt") as fo:
        json.dump(vocab[2:], fo, indent=4, sort_keys=True)


def read_dumped_vocab(vocab_dir):
    with open(os.path.join(vocab_dir, "vocab.json"), "rt") as fi:
        vocab = json.read(fo)
    return vocab


def dict_to_hyperparameters(hp_dict):
    return SimpleNamespace(**hp_dict)


def read_hyperparams(read_dir):
    with open(os.path.join(read_dir, "hyperparams.json"), "rt") as fi:
        hp_dict = json.load(fi)
    return dict_to_hyperparameters(hp_dict)


def load_glove_format_embs(
    fn, pad_token, unk_token, allowed_vocab_set, top_terms=100000
):
    fn = os.path.join(EMBEDDINGS_DIR, fn)
    ret_vocab = [pad_token, unk_token]
    ret_embeddings = []
    with open(fn, "rt") as fd:
        i = 0
        for line in fd:
            line = line.strip().split(" ")
            if line[0] not in allowed_vocab_set:
                continue
            ret_vocab.append(line[0])
            line_embeddings = [float(e) for e in line[1:]]
            if len(line_embeddings) != 200:
                raise ValueError("len of embeddings read =", len(line_embeddings))
            ret_embeddings.append(line_embeddings)
            i += 1
            if i == top_terms:
                break

    ret_vocab = np.array(ret_vocab)
    ret_embeddings = np.array(ret_embeddings)
    # embedding for unk_token initialized as the mean of all embeddings.
    mean_embedding = np.mean(ret_embeddings, axis=0, keepdims=True)
    # embedding for pad_token initialized as the zero vector.
    zero_embedding = np.zeros_like(mean_embedding)
    ret_embeddings = np.vstack((zero_embedding, mean_embedding, ret_embeddings))

    print("ret_vocab shape =", ret_vocab.shape)
    print("ret_embeddings shape =", ret_embeddings.shape)

    return ret_vocab, ret_embeddings

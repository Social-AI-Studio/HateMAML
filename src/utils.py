import json
import logging
import logging
import logging
import logging
import logging
import os
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.consts import DEST_DATA_PKL_DIR, EMBEDDINGS_DIR
from src.data.datasets import HFDataset

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def dump_hyperparams(dump_dir, hp_dict):
    with open(os.path.join(dump_dir, "hyperparams.json"), "wt") as fo:
        json.dump(hp_dict, fo, indent=4, sort_keys=True)


def dump_vocab(dump_dir, vocab):
    with open(os.path.join(dump_dir, "vocab.json"), "wt") as fo:
        json.dump(vocab[2:], fo, indent=4, sort_keys=True)


def read_dumped_vocab(vocab_dir):
    with open(os.path.join(vocab_dir, "vocab.json"), "rt") as fi:
        vocab = json.read(fi)
    return vocab


def dict_to_hyperparameters(hp_dict):
    return SimpleNamespace(**hp_dict)


def read_hyperparams(read_dir):
    with open(os.path.join(read_dir, "hyperparams.json"), "rt") as fi:
        hp_dict = json.load(fi)
    return dict_to_hyperparameters(hp_dict)


def load_glove_format_embs(fn, pad_token, unk_token, allowed_vocab_set, top_terms=100000):
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


def get_single_dataloader_from_split(config, split_name, dataset_name=None, lang=None, to_shuffle=False, batch_size=None):
    data_pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}_{split_name}.pkl")

    data_df = pd.read_pickle(data_pkl_path, compression=None)
    data_df = data_df.sample(frac=1)
    logger.info(f"picking {data_df.shape[0]} rows from `{data_pkl_path}`")

    if config.dataset_type == "bert":
        dataset = HFDataset(data_df, config.tokenizer, max_seq_len=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        shuffle=to_shuffle,  # default set to False for meta-training as fixed batch represent an uniqe task
        drop_last=False if split_name == "test" else True,
    )

    return dataloader


def get_dataloaders_from_split(config, split_name, dataset_name=None, lang=None, batch_size=None):
    if split_name == "train" and config.num_meta_samples:
        data_pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}_{config.num_meta_samples}_{split_name}.pkl")
    else:
        data_pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}_{split_name}.pkl")

    data_df = pd.read_pickle(data_pkl_path, compression=None)
    data_df = data_df.sample(frac=1)
    train_df, val_df = train_test_split(data_df, train_size=0.9, stratify=data_df["label"])
    logger.info(f"picking {train_df.shape[0]} rows for training set from `{data_pkl_path}`")
    logger.info(f"picking {val_df.shape[0]} rows for validation set from `{data_pkl_path}`")

    if config.dataset_type == "bert":
        train_dataset = HFDataset(train_df, config.tokenizer, max_seq_len=config.max_seq_len)
        val_dataset = HFDataset(val_df, config.tokenizer, max_seq_len=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=config.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=config.num_workers)

    return train_dataloader, val_dataloader


def get_collate_langs_dataloader(config, meta_langs, dsn_map):
    logger.info(f"Provided language set = {meta_langs}")
    split_name = "train"
    train_fnames, val_fnames = [], []
    for lang in meta_langs:
        dataset_name = dsn_map.get(lang)
        pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}{lang}_{config.num_meta_samples}_{split_name}.pkl")
        cur_data_df = pd.read_pickle(pkl_path, compression=None)
        train_df, val_df = train_test_split(cur_data_df, train_size=0.85, stratify=cur_data_df["label"])
        train_fnames.append(train_df)
        val_fnames.append(val_df)
    train_df = pd.concat(train_fnames)
    val_df = pd.concat(val_fnames)
    logger.info(f"picking {train_df.shape[0]} rows for training set")
    logger.info(f"picking {val_df.shape[0]} rows for validation set")

    if config.dataset_type == "bert":
        train_dataset = HFDataset(train_df, config.tokenizer, max_seq_len=config.max_seq_len)
        val_dataset = HFDataset(val_df, config.tokenizer, max_seq_len=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    return train_dataloader, val_dataloader


class SilverDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.data_dict[idx].items()}
        return item

    def __len__(self):
        return len(self.data_dict)


def report_memory(name=""):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | cached: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max cached: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string, end="\r")

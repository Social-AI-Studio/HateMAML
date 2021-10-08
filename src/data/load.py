from src.data.consts import DEST_DATA_PKL_DIR
from src.data.datasets import HFDataset, LSTMDataset
from math import ceil
import os
import pandas as pd
import torch


def get_dataloader(dataset_name, split_name, config):

    pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}_{split_name}.pkl")
    data_df = pd.read_pickle(pkl_path, compression=None)
    if config.dataset_type == "bert":
        dataset = HFDataset(
            data_df, config.tokenizer, max_seq_len=config.hp.max_seq_len
        )
    elif config.dataset_type == "lstm":
        dataset = LSTMDataset(
            df=data_df,
            vocab=config.vocab,
            max_seq_len=config.hp.max_seq_len,
            pad_token=config.pad_token,
            unk_token=config.unk_token,
        )
    else:
        raise ValueError(f"Unknown dataset_type {dataset_type}")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.hp.batch_size, num_workers=config.num_workers
    )

    return dataloader


def get_3_splits_dataloaders(dataset_name, config):
    """
    give all 3 splits' dataloaders for the `dataset_name` dataset.
    """

    split_names = ["train", "val", "test"]
    dataloaders = dict()

    for split_name in split_names:
        dataloaders[split_name] = get_dataloader(dataset_name, split_name, config)
    return dataloaders

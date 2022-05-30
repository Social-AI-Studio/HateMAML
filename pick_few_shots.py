import argparse
from math import ceil
import os
import pandas as pd

from src.data.consts import DEST_DATA_PKL_DIR


def process_src_pkl(
    shots, sampling, src_pkl_path, dest_pkl_path, force, lang=None, rng_seed=1
):
    dest_pkl_path = os.path.join(DEST_DATA_PKL_DIR, dest_pkl_path)
    src_pkl_path = os.path.join(DEST_DATA_PKL_DIR, src_pkl_path)

    if not os.path.isfile(src_pkl_path):
        raise FileNotFoundError(f"`{src_pkl_path}` path does not exist")
    if os.path.isfile(dest_pkl_path) and not force:
        raise FileExistsError(
            f"`{dest_pkl_path}` path already exists. Delete it or re-run this script with `--force` flag"
        )

    src_data_df = pd.read_pickle(src_pkl_path)

    if lang is not None:
        print(f"filtering only '{lang}' samples from the source pickle")
        src_data_df = src_data_df.query(f"lang == '{lang}'")

    src_labels = src_data_df.label.value_counts()
    src_labels_df = pd.concat(
        [src_labels, src_labels / src_labels.sum()], axis=1, names=["total", "ratio"]
    )
    print("label distribution before sampling:", src_labels_df)

    if sampling == "random":
        dest_data_df = src_data_df.sample(shots, random_state=rng_seed)
    elif sampling == "stratified":
        # strafication code picked from: https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas
        factor = shots / src_data_df.shape[0]
        dest_data_df = src_data_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(ceil(len(x) * factor), random_state=rng_seed)
        )
    elif sampling == "equal":
        num_labels = src_data_df.label.value_counts().shape[0]
        label_shots = shots // num_labels
        dest_data_df = src_data_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(label_shots, random_state=rng_seed)
        )
    elif sampling == "maximize":
        num_labels = src_data_df.label.value_counts().shape[0]
        label_shots = shots // num_labels
        mini = min(min(src_data_df.label.value_counts().to_list()), label_shots)
        label_shots = 50
        dest_data_df = src_data_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), shots-mini), random_state=rng_seed)
        )

    dest_labels = dest_data_df.label.value_counts()
    dest_labels_df = pd.concat(
        [dest_labels, dest_labels / dest_labels.sum()], axis=1, names=["total", "ratio"]
    )
    print("label distribution after sampling:", dest_labels_df)

    dest_data_df.to_pickle(dest_pkl_path, compression=None)
    print(
        f"successfully sampled {dest_data_df.shape[0]} comments from `{src_pkl_path}` to `{dest_pkl_path}`"
    )
    if dest_data_df.shape[0] != shots:
        print(
            f"WARNING: destination pickle has != {shots} shots: {dest_data_df.shape[0]}. This is possible because of various sampling techniques."
        )


def main(args):
    src_pkl_path = args["src_pkl"].strip()
    dest_pkl_path = args["dest_pkl"].strip()
    assert args["sampling"] in [
        "random",
        "stratified",
        "equal",
        "maximize",
    ], '`sampling` can only be one of: ["random","stratified","equal","maximize"]'
    process_src_pkl(
        shots=args["shots"],
        sampling=args["sampling"],
        src_pkl_path=src_pkl_path,
        dest_pkl_path=dest_pkl_path,
        force=args["force"],
        lang=args["lang"],
        rng_seed=args["rng_seed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_pkl",
        type=str,
        required=True,
        help="source _processed_ data pickle file name (WITHOUT the `data/processed/` prefix)",
    )
    parser.add_argument(
        "--dest_pkl",
        type=str,
        required=True,
        help="destination _processed_ data pickle file name (WITHOUT the `data/processed/` prefix)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        required=True,
        help="number of shots (/comments) to be sampled",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=1,
        help="RNG seed for the few-shot sampler",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        help='type of sampling to be done. possible value in ["random","stratified","equal"]',
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="if provided, only samples with this language code would be picked _before choosing the few shots_. helpful if a certain language from the src pickle is to be picked.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="if true, will force write destination pickle file",
    )

    args = parser.parse_args()
    args = vars(args)
    main(args)

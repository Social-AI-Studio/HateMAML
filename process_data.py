import argparse
import os
import pandas as pd

from src.data.consts import SRC_DATA_PKL_DIR, DEST_DATA_PKL_DIR
from src.data.preprocess import preprocess_text


def process_row(row):
    row.text = preprocess_text(row.text, row.lang)
    return row


def process_src_pkl(src_pkl_path, force, lang=None):
    dest_pkl_path = os.path.join(DEST_DATA_PKL_DIR, src_pkl_path)
    src_pkl_path = os.path.join(SRC_DATA_PKL_DIR, src_pkl_path)

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
    dest_data_df = src_data_df.apply(process_row, axis=1)

    dest_data_df.to_pickle(dest_pkl_path, compression=None)
    print(f"successfully processed `{src_pkl_path}` to `{dest_pkl_path}`")


def main(args):
    src_pkl_paths = args["src_pkl"].split(",")
    for src_pkl_path in src_pkl_paths:
        if src_pkl_path == "":
            continue
        process_src_pkl(src_pkl_path, args["force"], args["lang"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_pkl",
        type=str,
        required=True,
        help="comma-separated source data pickle file names (WITHOUT the `data/raw/` prefix)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="if provided, only samples with this language code would be processed. helpful if a certain language from the raw pickles are to be picked.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="if true, will force write processed pickle file(s)",
    )

    args = parser.parse_args()
    args = vars(args)
    main(args)

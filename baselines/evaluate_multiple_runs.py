import datetime
import glob
import json
import os
import sys

sys.path.append(".")

import argparse
import numpy as np

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from sklearn.metrics import f1_score

from src.config import EmptyConfig
from src.data.load import get_dataloader
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.model.lightning import LitClassifier
from src.utils import read_hyperparams


import torch
from transformers import AutoTokenizer


def find_best_checkpoint_from_dir(dir):
    checkpoint_paths = glob.glob(os.path.join(dir, "epoch=*.ckpt"))
    l = list()
    for checkpoint_path in checkpoint_paths:
        ckpt_name = os.path.split(checkpoint_path)[1]
        val_macro_f1_val = float(ckpt_name[-10:-5])
        l.append(
            (
                val_macro_f1_val,
                checkpoint_path,
            )
        )
    l.sort()
    print(f">>>found best checkpoint `{l[-1][1]}` with val_macro_f1_val=`{l[-1][0]}`")
    return l[-1][1]

def get_metrics(model,test_dataloader):
    model.eval()

    pred_labels_l = list()
    actual_labels_l = list()

    for batch_idx,batch in test_dataloader:
        batch_ret_dict = model.shared_step(batch,batch_idx,return_pred_labels=True)
        pred_labels_l.append(batch_ret_dict["pred_labels"].reshape(-1))
        actual_labels_l.append(batch_ret_dict["actual_labels"].reshape(-1))

    pred_labels = np.append(pred_labels_l)
    actual_labels = np.append(actual_labels_l)
    del pred_labels_l, actual_labels

    ret_metrics = {}
    ret_metrics["test_acc"] = (pred_labels == actual_labels).sum() / pred_labels.shape[0]
    ret_metrics["test_macro_f1"] = f1_score(actual_labels, pred_labels, average="macro")

    return ret_metrics

def main(args):
    assert args["model_type"] in [
        "xlmr",
        "mbert",
        "lstm",
    ], '`--model_type` argument must be in ["xlmr","mbert","lstm"]'

    if args["model_type"] == "lstm":
        raise NotImplementedError('not implemented for model_type == "lstm"')

    if args["model_type"] in ["xlmr", "mbert"]:
        dataset_type = "bert"
        if args["model_type"] == "xlmr":
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    else:
        raise NotImplementedError('not implemented for model_type == "lstm"')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args["model_type"] == "xlmr":
        model = XLMRClassifier(config=None)
    if args["model_type"] == "mbert":
        model = MBERTClassifier(config=None)
    elif args["model_type"] == "lstm":
        model = LSTMClassifier(config=None)

    lit_model = LitClassifier(model, config=None)
    lit_model.to(device)
    print(">>>moved model to device:", device)

    trainer = pl.Trainer(gpus=1)

    test_run_dirs = glob.glob(os.path.join(args["test_run_dirs_parent"], "*"))
    macro_f1_lists = {}
    acc_lists = {}
    for test_run_dir in test_run_dirs:
        if not os.path.isdir(test_run_dir):
            continue
        print(f">>>>test_run_dir = `{test_run_dir}`. finding best checkpoint now.")
        best_checkpoint = find_best_checkpoint_from_dir(test_run_dir)

        # config can be initialized with default instead of empty values.
        config = EmptyConfig()

        config.lang = args["lang"]
        config.num_workers = args["num_workers"]
        config.dataset_type = dataset_type
        config.tokenizer = tokenizer

        config.hp = read_hyperparams(test_run_dir)
        print(f">>>success loading hyperparams from path {test_run_dir}")

        print(">>>starting test")
        ckpt = torch.load(best_checkpoint)
        lit_model.load_state_dict(ckpt["state_dict"])
        for test_dataset_name in args["test_dataset_names"].split(","):
            print(">>>>testing on dataset:", test_dataset_name)
            test_dataloader = get_dataloader(test_dataset_name, "test", config)
            #test_results = trainer.test(
            #    model=lit_model, test_dataloaders=test_dataloader
            #)[0]
            test_results = get_metrics(lit_model,test_dataloader)
            print(
                f">>>>test_macro_f1 = {test_results['test_macro_f1']}, test_acc = {test_results['test_acc']}"
            )
            if test_dataset_name not in macro_f1_lists:
                macro_f1_lists[test_dataset_name] = []
            macro_f1_lists[test_dataset_name].append(test_results['test_macro_f1'])

            if test_dataset_name not in acc_lists:
                acc_lists[test_dataset_name] = []
            acc_lists[test_dataset_name].append(test_results['test_acc'])

    for test_dataset_name in args["test_dataset_names"].split(","):
        print(f">>>>>statistics for test dataset: {test_dataset_name}")
        mean_macro_f1 = np.mean(macro_f1_lists[test_dataset_name])
        std_macro_f1 = np.std(macro_f1_lists[test_dataset_name])
        print(f"{macro_f1_lists[test_dataset_name]}")
        print(f"mean macro f1: {mean_macro_f1}")
        print(f"std macro f1: {std_macro_f1}")
        mean_acc = np.mean(acc_lists[test_dataset_name])
        std_acc = np.std(acc_lists[test_dataset_name])
        print(f"{acc_lists[test_dataset_name]}")
        print(f"mean acc: {mean_acc}")
        print(f"std acc: {std_acc}")


if __name__ == "__main__":

    # parse commandline arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="should be one of: xlmr, mbert, lstm",
    )
    parser.add_argument(
        "--test_run_dirs_parent",
        type=str,
        default=None,
        help="parent directory of the runs to be evaluated on",
    )
    parser.add_argument(
        "--test_dataset_names",
        type=str,
        default=None,
        help="(comma separated) names of datasets to be tested on",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="provide if samples from `lang` language are to be filtered",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        # help="",
    )
    args = parser.parse_args()
    args = vars(args)
    print(">>>commandline args provided:\n", json.dumps(args, sort_keys=True, indent=4))

    main(args)

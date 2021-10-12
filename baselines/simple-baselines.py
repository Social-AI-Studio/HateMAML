import datetime
import os
import sys

sys.path.append(".")

import argparse
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from src.config import EmptyConfig
from src.data.consts import RUN_BASE_DIR, PAD_TOKEN, UNK_TOKEN
from src.data.load import get_3_splits_dataloaders
from src.model.classifiers import XLMRClassifier
from src.model.lightning import LitClassifier
from src.utils import dump_hyperparams, load_glove_format_embs, read_hyperparams


import torch
from transformers import AutoTokenizer


def main(args):
    assert (args["train"]) == (
        args["test_ckpt"] is None
    ), "Either set `--train` flag or provide `--test_ckpt` string argument, not both or none"

    assert args["model_type"] in [
        "xlmr",
        "mbert",
        "lstm",
    ], '`--model_type` argument must be in ["xlmr","mbert","lstm"]'

    assert (args["model_type"] == "lstm") == (
        args["embedding_txt_path"] is not None and args["hidden_dim"] is not None
    ), 'Provide `embedding_txt_path` and `hidden_dim` iff `model_type` == "lstm"'

    # config can be initialized with default instead of empty values.
    config = EmptyConfig()

    config.lang = args["lang"]

    if args["model_type"] in ["xlmr", "mbert"]:
        config.dataset_type = "bert"
        if args["model_type"] == "xlmr":
            config.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        else:
            raise NotImplementedError("yet to implement the mbert model.")
    else:
        config.dataset_type = "lstm"
    config.num_workers = args["num_workers"]

    if args["train"]:
        config.hp.batch_size = args["batch_size"]
        config.hp.lr = args["lr"]
        config.hp.es_patience = args["es_patience"]
        config.hp.epochs = args["epochs"]
        config.hp.max_seq_len = args["max_seq_len"]
        config.hp.dropout = args["dropout"]
        if args["model_type"] == "lstm":
            config.hp.embedding_txt_path = args["embedding_txt_path"]
            config.hp.hidden_dim = args["hidden_dim"]
        print("success reading hyperparams from arguments")
    else:
        hp_path = os.path.dirname(
            os.path.join(
                RUN_BASE_DIR, "baselines", args["model_type"], args["test_ckpt"]
            )
        )
        config.hp = read_hyperparams(hp_path)
        print(f"success loading hyperparams from path {hp_path}")

    if args["model_type"] == "lstm":
        config.pad_token, config.unk_token = PAD_TOKEN, UNK_TOKEN
        config.vocab, config.embeddings = load_glove_format_embs(
            config.hp.embedding_txt_path, config.pad_token, config.unk_token
        )

    dataloaders = get_3_splits_dataloaders(
        dataset_name=args["dataset_name"], config=config
    )
    for split_name in dataloaders.keys():
        print(
            f"successfully initialized {split_name} dataloader of {len(dataloaders[split_name])} batches"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args["rng_seed"])

    if args["model_type"] == "xlmr":
        model = XLMRClassifier(config)
    elif args["model_type"] == "lstm":
        model = LSTMClassifier(config)

    lit_model = LitClassifier(model, config)
    lit_model.to(device)
    print("moved model to device:", device)

    if args["train"]:
        print("starting train")

        lit_model.set_trainable(True)

        run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = os.path.join(RUN_BASE_DIR, "baselines", args["model_type"], run_name)
        os.mkdir(run_dir)
        dump_hyperparams(run_dir, vars(config.hp))

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=run_dir,
            filename="{epoch}-{val_acc:.3f}",
            save_last=True,
            save_top_k=5,
            monitor="val_macro_f1",
            mode="max",
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=config.hp.es_patience,
            verbose=False,
            mode="max",
        )

        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(run_dir, "logs"))
        tb_logger.log_hyperparams(vars(config.hp))

        trainer = pl.Trainer(
            gpus=1,
            # fast_dev_run=True,
            max_epochs=config.hp.epochs,
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
            ],
            logger=tb_logger,
            # precision=16,
        )

        trainer.fit(
            lit_model,
            train_dataloader=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
    else:
        print("starting test")
        ckpt_path = os.path.join(
            RUN_BASE_DIR, "baselines", args["model_type"], args["test_ckpt"]
        )
        ckpt = torch.load(ckpt_path)
        lit_model.load_state_dict(ckpt["state_dict"])
        trainer = pl.Trainer(gpus=1)
        trainer.test(model=lit_model, test_dataloaders=dataloaders["test"])


if __name__ == "__main__":

    # parse commandline arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="if provided, only samples with this language code would be used. helpful if certain languages from the pickles are to be used.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="should be one of: xlmr, mbert, lstm",
    )
    parser.add_argument(
        "--embedding_txt_path",
        type=str,
        default=None,
        help='provide iff `model_type` == "lstm"',
    )
    parser.add_argument(
        "--hidden_dim",
        type=str,
        default=None,
        help='provide iff `model_type` == "lstm"',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        # help="",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        # help="",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
        # help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        # help="",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        # help="",
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=3,
        # help="",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        # help="",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        # help="",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        # help="",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        # help="",
    )
    parser.add_argument(
        "--test_ckpt",
        type=str,
        default=None,
        help="if provided, this script will execute in testing mode.",
    )
    args = parser.parse_args()
    args = vars(args)

    main(args)

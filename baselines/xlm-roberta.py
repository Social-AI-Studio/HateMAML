import datetime
import os
import sys

sys.path.append(".")

import argparse
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from src.config import EmptyConfig
from src.data.consts import RUN_BASE_DIR
from src.data.load import get_3_splits_dataloaders
from src.model.lightning import LitClassifier
from src.utils import dump_hyperparams,read_hyperparams


import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification


def main(args):
    assert (args["train"]) == (args["test_ckpt"] is None), "Either set `--train` flag or provide `--test_ckpt` string argument, not both or none"

    # config can be initialized with default instead of empty values.
    config = EmptyConfig()

    config.dataset_type = "bert"
    config.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    config.num_workers = args["num_workers"]

    if args["train"]:
        config.hp.batch_size = args["batch_size"]
        config.hp.lr = args["lr"]
        config.hp.es_patience = args["es_patience"]
        config.hp.epochs = args["epochs"]
        config.hp.max_seq_len = args["max_seq_len"]
        print('success reading hyperparams from arguments')
    else:
        hp_path = os.path.dirname(os.path.join(RUN_BASE_DIR, "baselines", "xlm-roberta", args["test_ckpt"]))
        config.hp = read_hyperparams(hp_path)
        print(f'success loading hyperparams from path {hp_path}')

    dataloaders = get_3_splits_dataloaders(
        dataset_name=args["dataset_name"], config=config
    )
    for split_name in dataloaders.keys():
        print(
            f"successfully initialized {split_name} dataloader of {len(dataloaders[split_name])} batches"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args["rng_seed"])
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base")
    model.to(device)
    print("moved model to device:", device)

    lit_model = LitClassifier(model, config)

    if args["train"]:
        print("starting train")
        run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = os.path.join(RUN_BASE_DIR, "baselines", "xlm-roberta", run_name)
        os.mkdir(run_dir)
        dump_hyperparams(run_dir,vars(config.hp))

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=run_dir,
            filename="{epoch}-{val_acc:.3f}",
            save_last=True,
            save_top_k=3,
            monitor="val_acc",
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
        ckpt_path = os.path.join(RUN_BASE_DIR, "baselines", "xlm-roberta", args["test_ckpt"])
        ckpt = torch.load(ckpt_path)
        lit_model.load_state_dict(ckpt['state_dict'])
        trainer = pl.Trainer(gpus=1)
        trainer.test(model=lit_model,test_dataloaders=dataloaders["test"])


if __name__ == "__main__":

    # parse commandline arguments.
    parser = argparse.ArgumentParser()
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
        default=2e-3
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
        # help="",
    )
    args = parser.parse_args()
    args = vars(args)

    main(args)

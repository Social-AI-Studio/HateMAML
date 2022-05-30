import argparse
import json
import logging
import os

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from transformers import AutoTokenizer

from src.config import EmptyConfig
from src.data.consts import PAD_TOKEN, RUN_BASE_DIR, UNK_TOKEN
from src.data.load import build_vocabulary_from_train_split, get_3_splits_dataloaders
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.model.lightning import LitClassifier
from src.utils import (
    dump_hyperparams,
    dump_vocab,
    load_glove_format_embs,
    read_dumped_vocab,
    read_hyperparams,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")
logger.addHandler(logging.FileHandler("core.log"))


def main(args):
    assert (args["train"]) == (
        args["test_ckpt"] is None
    ), "Either set `--train` flag or provide `--test_ckpt` string argument, not both or none"

    if not args["train"] and args["freeze_layers"] is not None:
        raise AssertionError("`--freeze_layers` can be provided only with `--train` flag set")

    assert args["freeze_layers"] in [
        None,
        "embeddings",
        "top3",
        "top6",
    ], '`--freeze_layers` can only be in ["embeddings","top3","top6"]'

    assert (args["train_ckpt"] is None) or (
        args["test_ckpt"] is None
    ), "Can not provide both `--train_ckpt` and `--test_ckpt`"

    assert args["model_type"] in [
        "xlmr",
        "mbert",
        "lstm",
    ], '`--model_type` argument must be in ["xlmr","mbert","lstm"]'

    if args["test_ckpt"] is not None and args["model_type"] == "lstm":
        assert (
            args["load_vocab_from_test_ckpt"] is None
        ), 'need to provide `--load_vocab_from_test_ckpt` when testing on `--model_type` == "lstm"'
        if args["load_vocab_from_test_ckpt"].lower() == "true":
            args["load_vocab_from_test_ckpt"] = True
        elif args["load_vocab_from_test_ckpt"].lower() == "false":
            args["load_vocab_from_test_ckpt"] = False
        else:
            raise ValueError(
                f'unknown value received for `--load_vocab_from_test_ckpt` ({args["load_vocab_from_test_ckpt"]})'
            )

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
            config.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
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
        config.hp.train_ckpt = args["train_ckpt"]
        config.hp.rng_seed = args["rng_seed"]

        if args["model_type"] == "lstm":
            config.hp.embedding_txt_path = args["embedding_txt_path"]
            config.hp.hidden_dim = args["hidden_dim"]
        logger.info("success reading hyperparams from arguments")
    else:
        hp_path = os.path.dirname(os.path.join(RUN_BASE_DIR, "baselines", args["model_type"], args["test_ckpt"]))
        config.hp = read_hyperparams(hp_path)
        logger.info(f"success loading hyperparams from path {hp_path}")

    if args["model_type"] == "lstm":
        config.pad_token, config.unk_token = PAD_TOKEN, UNK_TOKEN
        if args["train"]:
            allowed_vocab_set = build_vocabulary_from_train_split(dataset_name, config, min_df=1)
            config.vocab, config.embeddings = load_glove_format_embs(
                config.hp.embedding_txt_path,
                config.pad_token,
                config.unk_token,
                allowed_vocab_set,
            )
            config.hp.embedding_dim = config.embeddings.shape[1]
        else:
            if args["load_vocab_from_test_ckpt"]:
                load_vocab_dir = os.path.join(RUN_BASE_DIR, "baselines", args["model_type"], args["test_ckpt"])
                logger.info("load_vocab_from_test_ckpt is True, loading vocabulary from ({load_vocab_dir})")
                allowed_vocab_set = set(read_dumped_vocab(load_vocab_dir))
            else:
                allowed_vocab_set = None
            config.vocab, config.embeddings = load_glove_format_embs(
                args["embedding_txt_path"],
                config.pad_token,
                config.unk_token,
                allowed_vocab_set,
            )
            assert (
                config.embeddings.shape[1] == config.hp.embedding_dim
            ), f"embedding dimension loaded for test is not same as the one used for train ({config.embeddings.shape[1]} != {config.hp.embedding_dim})"

    dataset_name = args["dataset_name"]
    if args["lang"]:
        dataset_name += args["lang"]
    if args["fewshot"]:
        dataset_name += "_200"

    logger.info(f"Dataset name is {dataset_name}")
    dataloaders = get_3_splits_dataloaders(
        dataset_name=dataset_name,
        train_few_dataset_name=None,
        config=config,
    )
    for split_name in dataloaders.keys():
        logger.info(f"Successfully initialized {split_name} dataloader of {len(dataloaders[split_name])} batches")

    seed_everything(args["rng_seed"], workers=True)

    if args["model_type"] == "xlmr":
        model = XLMRClassifier(config)
    if args["model_type"] == "mbert":
        model = MBERTClassifier(config)
    elif args["model_type"] == "lstm":
        model = LSTMClassifier(config)

    lit_model = LitClassifier(model, config)

    if args["train"]:

        if args["train_ckpt"] is not None:
            ckpt_path = os.path.join(
                RUN_BASE_DIR,
                "baselines",
                args["model_type"],
                args["dataset_name"],
                args["train_ckpt"],
            )
            logger.info(f"loading model from `{ckpt_path}`")
            ckpt = torch.load(ckpt_path)
            lit_model.load_state_dict(ckpt["state_dict"])

        logger.info("Starting training")

        lit_model.set_trainable(True)
        if args["model_type"] == "lstm" and args["freeze_layers"] != "embeddings":
            raise AssertionError(
                'For `--train` and `--model_type` == "lstm", `--freeze_layers` == "embeddings" is required'
            )
        lit_model.set_freeze_layers(args["freeze_layers"])

        # run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = os.path.join(
            RUN_BASE_DIR,
            "baselines",
            args["dataset_name"],
            args["lang"],
            "few" if args["fewshot"] else "full",
        )
        os.makedirs(run_dir, exist_ok=True)
        dump_hyperparams(run_dir, vars(config.hp))
        if args["model_type"] == "lstm":
            dump_vocab(run_dir, config.vocab)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=run_dir,
            filename=args["model_type"] + str(args["rng_seed"]),
            save_last=False,
            save_top_k=1,
            monitor="val_loss",
            mode="max",
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_macro_f1",
            min_delta=0.00,
            patience=config.hp.es_patience,
            verbose=False,
            mode="max",
        )

        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(run_dir, "logs"))
        tb_logger.log_hyperparams(vars(config.hp))

        trainer = pl.Trainer(
            strategy="ddp_find_unused_parameters_false",
            accelerator="gpu",
            devices="auto",
            max_epochs=config.hp.epochs,
            log_every_n_steps=15,
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
            ],
            logger=tb_logger,
        )

        trainer.fit(
            lit_model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
        test_splits = [i.strip() for i in args["test_splits"].strip().split(",")]
        if "val" in test_splits or "all" in test_splits:
            logger.info("testing on val split of dataset:")
            trainer.test(model=lit_model, dataloaders=dataloaders["val"])
        if "test" in test_splits or "all" in test_splits:
            logger.info("testing on test split of dataset:")
            if dataset_name == "evalita2020":
                results = trainer.test(model=lit_model, dataloaders=dataloaders["news_test"])
                os.makedirs(os.path.join(run_dir, "tweets", f"test-results-{args['rng_seed']}.json"))
                os.makedirs(os.path.join(run_dir, "news", f"test-results-{args['rng_seed']}.json"))
                with open(
                    os.path.join(run_dir, "tweets", f"test-results-{args['rng_seed']}.json"),
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)
                results = trainer.test(model=lit_model, dataloaders=dataloaders["tweets_test"])
                with open(
                    os.path.join(run_dir, "news", f"test-results-{args['rng_seed']}.json"),
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)
            else:
                results = trainer.test(model=lit_model, dataloaders=dataloaders["test"])
                with open(os.path.join(run_dir, f"test-results-{args['rng_seed']}.json"), "w") as f:
                    json.dump(results, f, indent=4)

    else:
        print("starting test")
        ckpt_path = os.path.join(RUN_BASE_DIR, "baselines", args["model_type"], args["test_ckpt"])
        ckpt = torch.load(ckpt_path)
        lit_model.load_state_dict(ckpt["state_dict"])
        trainer = pl.Trainer(gpus=2)
        test_splits = [i.strip() for i in args["test_splits"].strip().split(",")]
        if "train" in test_splits or "all" in test_splits:
            print("testing on train split of dataset:")
        if "val" in test_splits or "all" in test_splits:
            print("testing on val split of dataset:")
            trainer.test(model=lit_model, test_dataloader=dataloaders["val"])
        if "test" in test_splits or "all" in test_splits:
            print("testing on test split of dataset:")
            trainer.test(model=lit_model, test_dataloader=dataloaders["test"])


if __name__ == "__main__":

    # parse commandline arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="",
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
        help="If enabled model fine-tuning will be executed on provided training set",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="If enabled only 200 samples will be used for fine-tuning on training set",
    )
    parser.add_argument(
        "--train_ckpt",
        type=str,
        default=None,
        help="if provided, the classifier _encoder_ (not the head)  will be loaded from this checkpoint.",
    )
    parser.add_argument(
        "--test_ckpt",
        type=str,
        default=None,
        help="if provided, this script will execute in testing mode.",
    )
    parser.add_argument(
        "--load_vocab_from_test_ckpt",
        type=str,
        default=None,
        help="`True` or `False`",
    )
    parser.add_argument(
        "--test_splits",
        type=str,
        default=None,
        help="`train`,`val`,`test`,`all`, or a comma seperated combination of these values.",
    )
    parser.add_argument(
        "--freeze_layers",
        type=str,
        default=None,
        help="",
    )
    args = parser.parse_args()
    args = vars(args)

    main(args)

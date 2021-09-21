# An example baseline to give an idea how the project is structured.
import sys
import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from src.data.consts import LOG_BASE_DIR
from src.model.lightning import LitClassifier

sys.path.append(".")

from src.config import EmptyConfig
from src.data.load import get_3_splits_dataloaders
from transformers import AutoTokenizer, RobertaForSequenceClassification


def main(args):
    # config can be initialized with default instead of empty values.
    config = EmptyConfig()

    config.dataset_type = "bert"
    config.tokenizer = AutoTokenizer.from_pretrained("roberta-base-uncased")
    config.batch_size = args["batch_size"]
    config.max_seq_len = args["max_seq_len"]
    config.num_workers = args["num_workers"]

    dataloaders = get_3_splits_dataloaders(
        dataset_name=args["dataset_name"], config=config
    )
    for split_name in dataloaders.keys():
        print(
            f"successfully initialized {split_name} dataloader of {len(dataloaders[split_name])} batches"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args["rng_seed"])
    model = RobertaForSequenceClassification.from_pretrained("roberta-base-uncased")
    model.to(device)
    print("moved model to device:", device)

    lit_model = LitClassifier(model, config)

    run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(LOG_BASE_DIR, "baselines", "roberta", run_name)
    )
    tb_logger.log_hyperparams(args)

    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        max_epochs=args["epochs"],
        # callbacks=[],
        logger=tb_logger,
        # precision=16,
    )

    trainer.fit(
        lit_model,
        train_dataloader=dataloaders["train"],
        val_dataloaders=[dataloaders["val"], dataloaders["test"]],
    )


if __name__ == "__main__":

    # parse commandline arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        # help="",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        # help="",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=True,
        # help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        # help="",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
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
    args = parser.parse_args()
    args = vars(args)

    main(args)

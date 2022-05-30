"""
A prototype of two step meta learning algorithm for multilingual hate detection.
"""
import argparse
import os
import random
import datetime
import json
from statistics import mode
from typing_extensions import runtime
import numpy as np
import pandas as pd
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
from baselines.STCKA.utils.metrics import assess
from src.data.consts import DEST_DATA_PKL_DIR, RUN_BASE_DIR, SRC_DATA_PKL_DIR
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.model.lightning import LitClassifier
import torch
import torch.nn.functional as F
import learn2learn as l2l

from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
import sys
from transformers import (
    AutoConfig,
    AutoModel,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from src.data.datasets import HFDataset
from transformers import logging as tflog

tflog.set_verbosity_error()

run_suffix = datetime.datetime.now().strftime("%Y_%m_%d_%H")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f"runs/logs/logs_{run_suffix}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_helper():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="data/processed/",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--base_model_path",
        default="runs/semeval2020/Mbert2.ckpt",
        type=str,
        help="Path to fine-tunes base-model, load from checkpoint!",
    )
    parser.add_argument(
        "--dataset_type", default="bert", type=str, help="The input data type. It could take bert, lstm, gpt2 as input."
    )
    parser.add_argument(
        "--dataset_name",
        default="semeval2020",
        type=str,
        help="The name of the task to train selected in the list: [semeval, hateval, hasoc]",
    )
    # other parameters
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the mini batch")
    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        required=True,
        help="Languages to use for inital fine-tuning that produces the `base-model`.",
    )
    parser.add_argument("--aux_lang", type=str, default=None, help="Auxiliary languages to use for meta-training.")
    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        required=True,
        help="After finishing meta-training, meta-tuned model evaluation on the target language.",
    )
    parser.add_argument("--max_seq_len", type=int, default=128, help="The maximum sequence length of the inputs.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="The number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for tokenization.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of gpus to use.")
    parser.add_argument("--device_id", type=str, default="0", help="Gpu id to use.")
    parser.add_argument("--num_meta_iterations", type=int, default=10, help="Number of outer loop iteratins.")
    parser.add_argument("--meta_lr", type=float, default=2e-5, help="The outer loop meta update learning rate.")
    parser.add_argument(
        "--load_saved_base_model",
        default=False,
        action="store_true",
        help="Fine-tune base-model loading from a given checkpoint.",
    )
    parser.add_argument("--ddp", default=False, action="store_true", help="Use distributed parallel training.")
    parser.add_argument(
        "--overwrite_cache", default=False, action="store_true", help="Overwrite cached results for a run."
    )
    parser.add_argument(
        "--finetune_fewshot",
        type=str,
        default=None,
        help="Meta-adaptation (step 2) flag. Used in target language MAML training on only n samples where `n` = 200.",
    )
    parser.add_argument(
        "--num_meta_samples", type=int, default=200, help="Number of available samples for meta-training."
    )
    args = parser.parse_args()
    logger.info(args)
    return args


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def compute_loss_acc(logits, labels, criterion):
    preds = torch.argmax(logits, dim=1)
    loss = criterion(logits, labels)

    if torch.cuda.is_available():
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

    f1 = f1_score(preds, labels, average="macro")
    return loss, f1


def checkpoint_ft_model(args, model):
    run_dir = os.path.join(RUN_BASE_DIR, "ft", f"{args.dataset_name}{args.target_lang}")
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, args.model_name_or_path)
    logger.debug(f"Saving fine-tuned checkpoint to {output_dir}")
    torch.save(model.state_dict(), output_dir)


def evaluate(args, model, validation_dataloader, device, type="test"):
    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    pred_label = None
    target_label = None
    for batch in validation_dataloader:
        batch["labels"] = batch.pop("label")
        with torch.no_grad():
            batch = move_to(batch, device)
            outputs = model(batch)
            loss, _ = compute_loss_acc(outputs["logits"], batch["labels"], loss_fn)
            if args.n_gpu > 1:
                loss = loss.mean()
            logits = outputs["logits"]

            if pred_label is not None and target_label is not None:
                pred_label = torch.cat((pred_label, logits), 0)
                target_label = torch.cat((target_label, batch["labels"]))
            else:
                pred_label = logits
                target_label = batch["labels"]

            total_eval_loss += loss.item()

        nb_eval_steps += 1

    val_loss = total_eval_loss / len(validation_dataloader)
    _, p, r, f1 = assess(pred_label, target_label)

    log_str = type.capitalize() + "iction" if type == "pred" else type.capitalize() + "uation"

    if type == "pred":
        logger.info("*** Running Model {} **".format(log_str))
        logger.info(f"  Num examples = {len(validation_dataloader)*args.batch_size}")
        logger.info(f"  Loss = {val_loss:.3f}")
        logger.info(f"  F1 = {f1:.3f}")

    return f1, val_loss


def finetune(args, model, train_dataloader, eval_dataloader, test_dataloader, device):
    logger.debug("***** Running training *****")
    logger.debug(f"  Num examples = {len(train_dataloader)*args.batch_size}")
    logger.debug(f"  Num Epochs = {args.num_train_epochs}")
    logger.debug(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.debug(f"  Total optimization steps = {args.num_train_epochs * len(train_dataloader)}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = optim.AdamW(optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon)

    loss_fn = torch.nn.CrossEntropyLoss()

    eval_steps = 5 if args.finetune_fewshot == "few" else 50
    patience = 5

    nb_train_steps = 0
    min_macro_f1 = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            model.zero_grad()
            batch["labels"] = batch.pop("label")
            batch = move_to(batch, device)
            outputs = model(batch)
            loss, acc = compute_loss_acc(outputs["logits"], batch["labels"], loss_fn)
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            opt.step()
            total_train_loss += loss.item()
            total_train_acc += acc
            if (nb_train_steps + 1) % eval_steps == 0:
                avg_train_loss = total_train_loss / nb_train_steps
                logger.debug(f"  Epoch {epoch+1}, step {nb_train_steps+1}, training loss: {avg_train_loss:.3f} \n")
                nb_train_steps += 1

        f1, _ = evaluate(args, model, eval_dataloader, device, type="eval")
        if min_macro_f1 < f1:
            min_macro_f1 = f1
            checkpoint_ft_model(args, model)
            patience = 5
        else:
            patience -= 1

        if patience == 0:
            break

    logger.info("============================================================")
    logger.info(f"Evaluating fine-tuned model performance on test set. Training lang = {args.target_lang}")
    logger.info("============================================================")

    path_to_model = os.path.join(RUN_BASE_DIR, "ft", f"{args.dataset_name}{args.target_lang}", args.model_name_or_path)
    logger.info(f"Loading fine-tuned model from the checkpoint {path_to_model}")

    if "xlm-r" in args.model_name_or_path:
        model = XLMRClassifier()
    elif "bert" in args.model_name_or_path:
        model = MBERTClassifier()
    else:
        raise ValueError(f"Model type {args.model_name_or_path} is unknown.")

    ckpt = torch.load(os.path.normpath(path_to_model), map_location=device)
    model.load_state_dict(ckpt)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    f1, loss = evaluate(args, model, test_dataloader, device, type="pred")

    result = {"examples": len(train_dataloader) * args.batch_size, "f1": f1, "loss": loss}
    return result


def main(args, cuda=True):
    """
    Fine-tuning on standard training datasets
    """

    if args.ddp:
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank
        )

    summary_output_dir = os.path.join("runs/summary", args.dataset_name, "finetune")
    few_ft = "_" + args.finetune_fewshot if args.finetune_fewshot else ""
    f_base_model = args.base_model_path[-11:-5] if "bert" in args.base_model_path else args.base_model_path[-10:-5]
    f_samples = "_" + str(args.num_meta_samples) if args.num_meta_samples != 200 else ""
    summary_fname = os.path.join(
        summary_output_dir,
        f"{f_base_model}_{few_ft}{f_samples}_{args.target_lang}.json",
    )
    logger.info(f"Output fname = {summary_fname}")
    if os.path.exists(summary_fname) and not args.overwrite_cache:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    logger.info(f"device : {device}")
    logger.info(f"gpu : {args.n_gpu}")

    if "xlm-r" in args.model_name_or_path:
        model = XLMRClassifier()
    elif "bert" in args.model_name_or_path:
        model = MBERTClassifier()
    else:
        raise ValueError(f"Model type {args.model_name_or_path} is unknown.")

    lit_model = LitClassifier(model)
    ckpt = torch.load(os.path.normpath(args.base_model_path), map_location=device)
    lit_model.load_state_dict(ckpt["state_dict"])
    model = lit_model.model

    if torch.cuda.device_count() > 1:
        if args.ddp:
            model = nn.parallel.DistributedDataParallel(model)
        else:
            model = nn.DataParallel(model)

    model.to(device)

    target_lang_dataloaders = get_split_dataloaders(args, dataset_name=args.dataset_name, lang=args.target_lang)

    # **** THE FEW-SHOT FINE-TUNING STEP ****

    summary = {}
    if args.finetune_fewshot == "few":
        train_dataloader = get_dataloader(
            split_name="train",
            config=args,
            train_few_dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            train="standard",
        )
        eval_dataloader = get_dataloader(
            split_name="val",
            config=args,
            train_few_dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            train="standard",
        )
        test_dataloader = target_lang_dataloaders["test"]
        ft_result = finetune(args, model, train_dataloader, eval_dataloader, test_dataloader, device)
        summary[str(args.finetune_fewshot)] = ft_result

    elif args.finetune_fewshot == "full":
        ft_result = finetune(
            args,
            model,
            target_lang_dataloaders["train"],
            target_lang_dataloaders["val"],
            target_lang_dataloaders["test"],
            device,
        )
        summary[str(args.finetune_fewshot)] = ft_result

    os.makedirs(summary_output_dir, exist_ok=True)
    json.dump(summary, open(summary_fname, "w"))


def get_split_dataloaders(config, dataset_name, lang):
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    split_names = ["train", "val", "test"]
    dataloaders = dict()

    for split_name in split_names:
        pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}{lang}_{split_name}.pkl")
        logger.debug(pkl_path)
        data_df = pd.read_pickle(pkl_path, compression=None)
        if lang is not None:
            logger.debug(f"filtering only '{lang}' samples from {split_name} pickle")
            data_df = data_df.query(f"lang == '{lang}'")
        if config.dataset_type == "bert":
            dataset = HFDataset(data_df, config.tokenizer, max_seq_len=config.max_seq_len)
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        if args.ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        split_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            sampler=sampler if args.ddp else None,
            pin_memory=True,
        )

        dataloaders[split_name] = split_dataloader
    return dataloaders


def get_dataloader(split_name, config, train_few_dataset_name=None, lang=None, train="meta"):
    if lang == "en":
        few_pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{train_few_dataset_name}_200_{split_name}.pkl")
    else:
        few_pkl_path = os.path.join(
            DEST_DATA_PKL_DIR, f"{train_few_dataset_name}_{config.num_meta_samples}_{split_name}.pkl"
        )
    data_df = pd.read_pickle(few_pkl_path, compression=None)
    logger.debug(f"picking {data_df.shape[0]} rows from `{few_pkl_path}` as few samples")

    if lang is not None:
        logger.debug(f"filtering only '{lang}' samples from {split_name} pickle")
        data_df = data_df.query(f"lang == '{lang}'")

    if config.dataset_type == "bert":
        dataset = HFDataset(data_df, config.tokenizer, max_seq_len=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")

    if train == "meta":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.shots,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=True if split_name == "train" else False,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True if split_name == "train" else False,
        )

    return dataloader


if __name__ == "__main__":
    args = parse_helper()
    main(args, cuda=True)

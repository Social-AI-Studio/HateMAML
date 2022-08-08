"""
A script for standard finetuning
"""
import argparse
import json
import logging
import math
import os
import random

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mtl_datasets import DataLoaderWithTaskname, MultitaskDataloader
from src.data.consts import DEST_DATA_PKL_DIR, RUN_BASE_DIR, SRC_DATA_PKL_DIR
from src.maml.mtl_datasets import DataLoaderWithTaskname
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.model.lightning import LitClassifier
from src.utils import get_dataloaders_from_split, get_single_dataloader_from_split


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
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--base_model_path",
        default=None,
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
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
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
        "--finetune_type",
        type=str,
        default=None,
        help="Finetuning choice. Used in target language MAML training on only n samples where `n` = 200.",
    )
    parser.add_argument(
        "--num_meta_samples", type=int, default=None, help="Number of available samples for meta-training."
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
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


def evaluate(args, model, eval_dataloader, device, type="test"):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    pred_label = []
    target_label = []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = move_to(batch, device)
            outputs = model(**batch)
            logits = outputs["logits"]
            pred_label.append(logits)
            target_label.append(batch["labels"])

    pred_label = torch.cat(pred_label)
    target_label = torch.cat(target_label)
    loss, f1 = compute_loss_acc(pred_label, target_label, loss_fn)
    loss = loss.item()
    log_str = "prediction" if type == "pred" else "evaluation"
    logger.info("*** Running {} **".format(log_str))
    logger.info(f"  Num examples = {len(eval_dataloader)*args.batch_size}")
    logger.info(f"  Loss = {loss:.3f}")
    logger.info(f"  F1 = {f1:.3f}")

    return f1, loss


def finetune(args, model, train_dataloader, eval_dataloader, test_dataloader, device):
    output_dir = os.path.join(
        RUN_BASE_DIR, "finetune", args.dataset_name, args.target_lang, args.finetune_type, f"seed{args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
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

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=opt, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_training_steps
    )
    eval_steps = 50
    nb_train_steps = 0
    min_loss = 1000

    for epoch in tqdm(range(args.num_train_epochs), desc="Iteration"):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Batch"):
            model.zero_grad()
            if "task_name" in batch.keys():
                batch.pop("task_name")
            batch = move_to(batch, device)
            outputs = model(**batch)
            loss, acc = compute_loss_acc(outputs["logits"], batch["labels"], loss_fn)
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            opt.step()
            lr_scheduler.step()
            total_train_loss += loss.item()
            total_train_acc += acc
            nb_train_steps += 1
            if (nb_train_steps + 1) % eval_steps == 0:
                avg_train_loss = total_train_loss / nb_train_steps
                logger.info(f"  Epoch {epoch+1}, step {nb_train_steps+1}, training loss: {avg_train_loss:.3f} \n")

        f1, loss = evaluate(args, model, eval_dataloader, device, type="eval")
        if min_loss > loss:
            min_loss = loss
            logger.info(f"Saving fine-tuned checkpoint from epoch {epoch+1} at {output_dir}")
            model.save_pretrained(output_dir)

    logger.info(f"Loading fine-tuned model from the checkpoint {output_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    logger.info("============================================================")
    logger.info(f"Evaluating fine-tuned model performance on test set. Training lang = {args.target_lang}")
    logger.info("============================================================")
    f1, loss = evaluate(args, model, test_dataloader, device, type="pred")
    result = {"examples": len(test_dataloader) * args.batch_size, "f1": f1, "loss": loss}
    return result


def main(args, cuda=True):
    """
    Fine-tuning on standard training datasets
    """

    if args.ddp:
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank
        )

    summary_output_dir = os.path.join("runs/finetune", args.dataset_name, args.target_lang)
    few_ft = "_" + args.finetune_type if args.finetune_type else ""
    f_base_model = "mbert" if "bert" in args.model_name_or_path else "xlmr"
    f_samples = "_" + str(args.num_meta_samples) if args.num_meta_samples else ""
    summary_fname = os.path.join(
        summary_output_dir,
        f"{f_base_model}{args.seed}{few_ft}{f_samples}.json",
    )
    logger.info(f"Output fname = {summary_fname}")
    if os.path.exists(summary_fname) and not args.overwrite_cache:
        return

    seed = args.seed
    logger.info(f"Setting up seed value {seed}")
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

    args.src_dataset_name = "hasoc2020" if args.dataset_name == "semeval2020" else args.dataset_name
    # fft requires a base model
    if args.finetune_type in ["fft", "zero"] and args.base_model_path is None:
        raise ValueError(f"The field args.base_model_path is empty while finetune type {args.finetune_type}")

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.base_model_path:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    if torch.cuda.device_count() > 1:
        if args.ddp:
            model = nn.parallel.DistributedDataParallel(model)
        else:
            model = nn.DataParallel(model)

    model.to(device)

    # **** THE FEW-SHOT FINE-TUNING STEP ****

    summary = {}
    if args.finetune_type in ["ft", "fft"]:
        dataloaders = {}
        for split in ["train", "val", "test"]:
            add_to_path = f"_{args.num_meta_samples}" if args.num_meta_samples and split == "train" else ""
            dataloader = get_single_dataloader_from_split(
                config=args,
                split_name=split,
                dataset_name=f"{args.dataset_name}{args.target_lang}" + add_to_path,
                lang=args.target_lang,
                to_shuffle=True  if split == "train" else False, 
                batch_size=args.batch_size,
            )
            dataloaders[split] = dataloader
        ft_result = finetune(args, model, dataloaders["train"], dataloaders["val"], dataloaders["test"], device)
        summary[str(args.finetune_type)] = ft_result

    # fine-tuning only using English data
    elif args.finetune_type == "en_ft":
        train_dataloader = get_single_dataloader_from_split(
            config=args,
            split_name="train",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            batch_size=args.batch_size,
        )
        eval_dataloader = get_single_dataloader_from_split(
            config=args,
            split_name="val",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            batch_size=args.batch_size,
        )
        test_dataloader = get_single_dataloader_from_split(
            config=args,
            split_name="test",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            batch_size=args.batch_size,
        )

        ft_result = finetune(args, model, train_dataloader, eval_dataloader, test_dataloader, device)
        summary[str(args.finetune_type)] = ft_result

    elif args.finetune_type == "ft_en":
        dataloaders = {}
        dataloaders[args.source_lang] = DataLoaderWithTaskname(
            task_name=args.source_lang,
            data_loader=get_single_dataloader_from_split(
                config=args,
                split_name="train",
                dataset_name=f"{args.src_dataset_name}{args.source_lang}",
                lang=args.source_lang,
                to_shuffle=True,
                batch_size=args.batch_size,
            ),
        )
        dataloaders[args.target_lang], eval_dataloader = get_dataloaders_from_split(
            config=args,
            split_name="val",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            batch_size=args.batch_size,
        )
        train_dataloader = MultitaskDataloader(dataloaders)
        # eval_dataloader = get_single_dataloader_from_split(
        #     config=args,
        #     split_name="val",
        #     dataset_name=f"{args.src_dataset_name}{args.source_lang}",
        #     lang=args.target_lang,
        #     batch_size=args.batch_size,
        # )
        test_dataloader = get_single_dataloader_from_split(
            config=args,
            split_name="test",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            batch_size=args.batch_size,
        )

        ft_result = finetune(args, model, train_dataloader, eval_dataloader, test_dataloader, device)
        summary[str(args.finetune_type)] = ft_result
    elif args.finetune_type == "zero":
        test_dataloader = get_single_dataloader_from_split(
            config=args,
            split_name="val",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            batch_size=args.batch_size,
        )

        logger.info(f"**** Zero-shot evaluation on {args.dataset_name}  >>> Language = {args.target_lang} ****")
        f1, loss = evaluate(args, model, test_dataloader, device, type="pred")
        zero_result = {"examples": len(test_dataloader) * args.batch_size, "f1": f1, "loss": loss}
        summary[str(args.finetune_type)] = zero_result

    os.makedirs(summary_output_dir, exist_ok=True)
    json.dump(summary, open(summary_fname, "w"))


if __name__ == "__main__":
    args = parse_helper()
    main(args, cuda=True)

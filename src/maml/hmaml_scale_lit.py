"""
A prototype of two step meta learning algorithm for multilingual hate detection.
"""
from statistics import mode
import sys
import os
import argparse
import random
import datetime
import json
import numpy as np
import pandas as pd
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
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f"runs/logs/logs_{run_suffix}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
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
        help="Path to pre-trained model or shortcut name selected in the list: `mbert`, `xlm-r`",
    )
    parser.add_argument(
        "--base_model_path",
        default="runs/tanmoy/Mbert2.ckpt",
        type=str,
        help="Path to fine-tunes base-model, load from checkpoint!",
    )
    parser.add_argument(
        "--dataset_type",
        default="bert",
        type=str,
        help="The input data type. It could take bert, lstm, gpt2 as input.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )

    parser.add_argument("--shots", default=8, type=int, help="Size of the mini batch")
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Size of the mini batch"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="The maximum sequence length of the inputs.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="The number of training epochs."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for tokenization."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of gpus to use.")
    parser.add_argument("--device_id", type=str, default="0", help="Gpu id to use.")
    parser.add_argument(
        "--num_meta_iterations",
        type=int,
        default=10,
        help="Number of outer loop iteratins.",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=2e-5,
        help="The outer loop meta update learning rate.",
    )
    parser.add_argument(
        "--fast_lr",
        type=float,
        default=4e-5,
        help="The inner loop fast adaptation learning rate.",
    )
    parser.add_argument(
        "--load_saved_base_model",
        default=False,
        action="store_true",
        help="Fine-tune base-model loading from a given checkpoint.",
    )
    parser.add_argument(
        "--overwrite_cache",
        default=False,
        action="store_true",
        help="Overwrite cached results for a run.",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="maml-step1",
        help="Provide a MAML training setup. Valid values are 'hmaml_scale'.",
    )
    parser.add_argument(
        "--num_meta_samples",
        type=int,
        default=200,
        help="Number of available samples for meta-training.",
    )
    parser.add_argument(
        "--freeze_layers",
        type=str,
        default=None,
        help='Set freeze layers. `freeze_layers` can only be in ["embeddings","top3","top6"]',
    )
    parser.add_argument(
        "--meta_langs",
        type=str,
        default="",
        help="List of languages to support during meta training or model fine-tuning.",
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
    run_dir = os.path.join(RUN_BASE_DIR, "ft", f"{args.exp_setting}")
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

    log_str = (
        type.capitalize() + "iction" if type == "pred" else type.capitalize() + "uation"
    )

    if type == "pred":
        logger.info("*** Running Model {} **".format(log_str))
        logger.info(f"  Num examples = {len(validation_dataloader)*args.batch_size}")
        logger.info(f"  Loss = {val_loss:.3f}")
        logger.info(f"  F1 = {f1:.3f}")

    return f1, val_loss


def finetune(
    args,
    model,
    train_dataloader,
    eval_dataloader,
    test_dataloader_list,
    meta_langs,
    device,
):
    logger.debug("***** Running training *****")
    logger.debug(f"  Num examples = {len(train_dataloader)*args.batch_size}")
    logger.debug(f"  Num Epochs = {args.num_train_epochs}")
    logger.debug(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.debug(
        f"  Total optimization steps = {args.num_train_epochs * len(train_dataloader)}"
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = optim.AdamW(
        optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    eval_steps = 50
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
                logger.debug(
                    f"  Epoch {epoch+1}, step {nb_train_steps+1}, training loss: {avg_train_loss:.3f} \n"
                )
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
    logger.info("============================================================")

    path_to_model = os.path.join(
        RUN_BASE_DIR,
        "ft",
        f"{args.exp_setting}",
        args.model_name_or_path,
    )
    logger.info(f"Loading fine-tuned model from the checkpoint {path_to_model}")

    if "xlm-r" in args.model_name_or_path:
        model = XLMRClassifier()
    elif "bert" in args.model_name_or_path:
        model = MBERTClassifier()
    else:
        raise ValueError(f"Model type {args.model_name_or_path} is unknown.")

    ckpt = torch.load(os.path.normpath(path_to_model), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    summary = {
        "meta_langs": meta_langs,
        "num_meta_iterations": args.num_meta_iterations,
        "base_model_path": args.base_model_path,
        "script_name": os.path.basename(__file__),
        "exp_setting": args.exp_setting,
    }

    for lang_id in meta_langs:
        logger.info(
            f"Evaluating fine-tuned model performance on test set. Lang = {lang_id}"
        )
        if lang_id == "it":
            for lg_id in ["news", "tweets"]:
                eval_result = evaluate(
                    args, model, test_dataloader_list[lg_id], device, type="pred"
                )
                summary[lg_id] = {"f1": eval_result[0], "loss": eval_result[1]}
        else:
            eval_result = evaluate(
                args, model, test_dataloader_list[lang_id], device, type="pred"
            )
            summary[lang_id] = {"f1": eval_result[0], "loss": eval_result[1]}
    return summary


def main(args, meta_batch_size=None, adaptation_steps=1, cuda=False, seed=42):
    """
    An implementation of two-step *Model-Agnostic Meta-Learning* algorithm for hate detection
    on low-resouce languages.

    Args:
        meta_lr (float): The learning rate used to update the model.
        fast_lr (float): The learning rate used to update the MAML inner loop.
        adaptation_steps (int); The number of inner loop steps.
        num_iterations (int): The total number of iteration MAML will run (outer loop update).
    """

    summary_output_dir = os.path.join(
        "runs/summary", "analyze", os.path.basename(__file__)[:-3]
    )
    f_base_model = (
        args.base_model_path[-11:-5]
        if "bert" in args.base_model_path
        else args.base_model_path[-10:-5]
    )
    meta_langs = args.meta_langs.split(",")
    # meta_langs = ["ar", "da", "gr", "tr", "hi", "de", "es"]
    etxt = (
        "_" + str(args.num_meta_iterations) if args.exp_setting == "hmaml-scale" else ""
    )
    summary_fname = os.path.join(
        summary_output_dir,
        f"{f_base_model}_{args.exp_setting}{etxt}_{len(meta_langs)}.json",
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
    logger.debug(f"device : {device}")
    logger.debug(f"gpu : {args.n_gpu}")

    if "xlm-r" in args.model_name_or_path:
        model = XLMRClassifier()
    elif "bert" in args.model_name_or_path:
        model = MBERTClassifier()
    else:
        raise ValueError(f"Model type {args.model_name_or_path} is unknown.")

    lit_model = LitClassifier(model)
    ckpt = torch.load(os.path.normpath(args.base_model_path), map_location=device)
    lit_model.load_state_dict(ckpt["state_dict"])

    if args.freeze_layers:
        lit_model.set_trainable(True)
        lit_model.set_freeze_layers(args.freeze_layers)
        for name, param in lit_model.model.named_parameters():
            logger.debug(
                "%s - %s", name, ("Unfrozen" if param.requires_grad else "FROZEN")
            )

    model = lit_model.model
    model.to(device)

    dsn_map = {
        "en": "founta",
        "es": "hateval2019",
        "ar": "semeval2020",
        "da": "semeval2020",
        "gr": "semeval2020",
        "tr": "semeval2020",
        "it": "evalita2020",
        "hi": "hasoc2020",
        "de": "hasoc2020",
    }
    if args.exp_setting == "hmaml-scale":
        meta_tasks_list = []
        for lang_id in meta_langs:
            logger.info(f"Loading meta tasks for {lang_id}")
            cur_dataloader = get_dataloader(
                split_name="val" if lang_id == "en" else "train",
                config=args,
                train_few_dataset_name=f"{dsn_map.get(lang_id)}{lang_id}",
                lang=lang_id,
            )
            meta_tasks_list.append(cur_dataloader)
        meta_batch_size_list = []
        for l in meta_tasks_list:
            meta_batch_size_list.append(len(l))
        meta_batch_size = sum(meta_batch_size_list)
        logger.info(f"Number of meta tasks {meta_batch_size}")

        meta_domain_tasks = meta_tasks_list[:]
    elif args.exp_setting == "finetune-collate":
        collate_dataloaders_list = collate_langs_dataloader(args, meta_langs, dsn_map)
    else:
        raise ValueError(f"{args.exp_setting} is unknown!")

    test_dataloader_list = {}
    for lang_id in meta_langs:
        logger.debug(
            f"**** Zero-shot evaluation on {dsn_map.get(lang_id)} before MAML >>> Language = {lang_id} ****"
        )
        if lang_id == "it":
            evalita_loaders = get_evalita_loaders(
                split_name="test", config=args, dataset_name=dsn_map.get(lang_id)
            )
            for kkey in evalita_loaders:
                test_dataloader_list[kkey] = evalita_loaders[kkey]
                # evaluate(args, mode, evalita_loaders[kkey], device, type="pred")
        else:
            cur_test_dataloader = get_dataloader(
                split_name="test",
                config=args,
                train_few_dataset_name=f"{dsn_map.get(lang_id)}{lang_id}",
                lang=lang_id,
                train="standard",
            )
            test_dataloader_list[lang_id] = cur_test_dataloader
            # evaluate(args, model, cur_test_dataloader, device, type="pred")

    if args.exp_setting == "finetune-collate":
        summary = finetune(
            args,
            model,
            collate_dataloaders_list["train"],
            collate_dataloaders_list["val"],
            test_dataloader_list,
            meta_langs,
            device,
        )
        logger.info(summary)
        os.makedirs(summary_output_dir, exist_ok=True)
        json.dump(summary, open(summary_fname, "w"))

        return

    # MAML training starts here. This step assumes we have a pretrained base model, fine-tuned on English.
    # We will now meta-train the base model with MAML algorithm.
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = optim.AdamW(optimizer_grouped_parameters, args.meta_lr, eps=args.adam_epsilon)

    logger.info("*********************************")
    logger.info("*** MAML training starts now ***")
    logger.info("*********************************")

    loss_fn = torch.nn.CrossEntropyLoss()

    for iteration in tqdm(range(args.num_meta_iterations)):  # outer loop
        opt.zero_grad()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_domain_accuracy = 0.0

        tmp_cntr = meta_batch_size_list[:]

        for idx in range(meta_batch_size):
            task = None
            choice_idx = None
            for k in range(10):
                choice_idx = random.randint(0, len(meta_tasks_list) - 1)
                if tmp_cntr[choice_idx] > 0:
                    task = next(iter(meta_tasks_list[choice_idx]))
                    tmp_cntr[choice_idx] -= 1
                    break
            if task is None:
                continue
            n_meta_lr = args.shots // 2

            train_query_inp = {
                "input_ids": task["input_ids"][:n_meta_lr],
                "attention_mask": task["attention_mask"][:n_meta_lr],
                "labels": task["label"][:n_meta_lr],
            }
            train_support_inp = {
                "input_ids": task["input_ids"][n_meta_lr:],
                "attention_mask": task["attention_mask"][n_meta_lr:],
                "labels": task["label"][n_meta_lr:],
            }

            train_query_inp = move_to(train_query_inp, device)
            train_support_inp = move_to(train_support_inp, device)

            learner = maml.clone()
            # report_memory(name = f"maml{idx}")

            for _ in range(adaptation_steps):
                outputs = learner(train_support_inp)
                loss, acc = compute_loss_acc(
                    outputs["logits"], train_support_inp["labels"], loss_fn
                )

                if args.n_gpu > 1:
                    loss = loss.mean()
                learner.adapt(loss, allow_nograd=True, allow_unused=True)
                meta_train_error += loss.item()
                meta_train_accuracy += acc
            outputs = learner(train_query_inp)
            eval_loss, eval_acc = compute_loss_acc(
                outputs["logits"], train_query_inp["labels"], loss_fn
            )

            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            meta_valid_error += eval_loss.item()
            meta_valid_accuracy += eval_acc

            if args.exp_setting in ["hmaml-scale"]:
                for k in range(10):
                    d_idx = random.randint(0, len(meta_domain_tasks) - 1)
                    if d_idx != choice_idx:
                        break
                b_idx = random.randint(0, len(meta_domain_tasks[d_idx]))

                for indx, d_batch in enumerate(meta_domain_tasks[d_idx]):
                    if indx == b_idx:
                        d_task = d_batch
                        break
                domain_query_inp = {
                    "input_ids": d_task["input_ids"][:n_meta_lr],
                    "attention_mask": d_task["attention_mask"][:n_meta_lr],
                    "labels": d_task["label"][:n_meta_lr],
                }
                domain_query_inp = move_to(domain_query_inp, device)
                outputs = learner(domain_query_inp)
                d_loss, d_f1 = compute_loss_acc(
                    outputs["logits"], domain_query_inp["labels"], loss_fn
                )
                if args.n_gpu > 1:
                    d_loss = d_loss.mean()

                total_loss = eval_loss + d_loss
                meta_domain_accuracy += d_f1

            else:
                total_loss = eval_loss

            total_loss.backward()

        # Print some metrics
        print("\n")
        mt_error = meta_train_error / meta_batch_size
        mt_acc = meta_train_accuracy / meta_batch_size
        mv_error = meta_valid_error / meta_batch_size
        mv_acc = meta_valid_accuracy / meta_batch_size
        md_acc = meta_domain_accuracy / meta_batch_size
        # mv_error.backward()

        logger.info(
            "  Iteration {} >> Meta Train Error {:.3f}, F1 {:.3f} # Valid Error {:.3f}, F1 {:.3f}, Domain F1 {:.3f}".format(
                iteration, mt_error, mt_acc, mv_error, mv_acc, md_acc
            )
        )

        # Average the accumulated gradients and optimize
        # for p in maml.parameters():
        #     if p.grad is not None:
        #         p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    # Zero-shot, Few-shot or Full-tuned evaluation? You can now take the zero-shot meta-trained model and fine-tune it on
    # available low-resource few-shot training samples. If we don't fine-tune further, it can be considered as zero-shot model.
    #
    summary = {
        "meta_langs": meta_langs,
        "num_meta_iterations": args.num_meta_iterations,
        "base_model_path": args.base_model_path,
        "script_name": os.path.basename(__file__),
        "exp_setting": args.exp_setting,
    }

    for lang_id in meta_langs:
        logger.info(
            f"**** Evaluation on {dsn_map.get(lang_id)} meta {args.exp_setting} tuned model >>> Language = {lang_id} ****"
        )
        if lang_id == "it":
            for lg_id in ["news", "tweets"]:
                eval_result = evaluate(
                    args, model, test_dataloader_list[lg_id], device, type="pred"
                )
                summary[lg_id] = {"f1": eval_result[0], "loss": eval_result[1]}
        else:
            eval_result = evaluate(
                args, model, test_dataloader_list[lang_id], device, type="pred"
            )
            summary[lang_id] = {"f1": eval_result[0], "loss": eval_result[1]}

    # **** THE FEW-SHOT FINE-TUNING STEP ****

    os.makedirs(summary_output_dir, exist_ok=True)
    json.dump(summary, open(summary_fname, "w"))


def get_split_dataloaders(config, dataset_name, lang):
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    split_names = ["train", "val", "test"]
    dataloaders = dict()

    for split_name in split_names:
        pkl_path = os.path.join(
            DEST_DATA_PKL_DIR, f"{dataset_name}{lang}_{split_name}.pkl"
        )
        logger.debug(pkl_path)
        data_df = pd.read_pickle(pkl_path, compression=None)
        if lang is not None:
            logger.debug(f"filtering only '{lang}' samples from {split_name} pickle")
            data_df = data_df.query(f"lang == '{lang}'")
        if config.dataset_type == "bert":
            dataset = HFDataset(
                data_df, config.tokenizer, max_seq_len=config.max_seq_len
            )
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        dataset = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True,
        )

        dataloaders[split_name] = dataset
    return dataloaders


def get_evalita_loaders(split_name, config, dataset_name):
    evalita_loaders = {}
    topics = ["tweets", "news"]
    for lang_id in topics:
        print(dataset_name)
        few_pkl_path = os.path.join(
            DEST_DATA_PKL_DIR,
            f"{dataset_name}{lang_id}_{split_name}.pkl",
        )
        data_df = pd.read_pickle(few_pkl_path, compression=None)
        logger.debug(
            f"picking {data_df.shape[0]} rows from `{few_pkl_path}` as few samples"
        )

        if config.dataset_type == "bert":
            dataset = HFDataset(
                data_df, config.tokenizer, max_seq_len=config.max_seq_len
            )
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        evalita_loaders[lang_id] = dataloader
    return evalita_loaders


def get_dataloader(
    split_name, config, train_few_dataset_name=None, lang=None, train="meta"
):
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if split_name == "test":
        few_pkl_path = os.path.join(
            DEST_DATA_PKL_DIR,
            f"{train_few_dataset_name}_{split_name}.pkl",
        )
    else:
        if lang == "en":
            few_pkl_path = os.path.join(
                DEST_DATA_PKL_DIR, f"{train_few_dataset_name}_200_{split_name}.pkl"
            )
        else:
            few_pkl_path = os.path.join(
                DEST_DATA_PKL_DIR,
                f"{train_few_dataset_name}_{config.num_meta_samples}_{split_name}.pkl",
            )

    data_df = pd.read_pickle(few_pkl_path, compression=None)
    logger.debug(
        f"picking {data_df.shape[0]} rows from `{few_pkl_path}` as few samples"
    )

    if lang is not None:
        logger.debug(f"filtering only '{lang}' samples from {split_name} pickle")
        data_df = data_df.query(f"lang == '{lang}'")

    if config.dataset_type == "bert":
        dataset = HFDataset(data_df, config.tokenizer, max_seq_len=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")

    if train == "meta":
        dataloader = DataLoader(
            dataset,
            batch_size=config.shots,
            num_workers=config.num_workers,
            drop_last=True,
            # shuffle=True if split_name == "train" else False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            # shuffle=True if split_name == "train" else False,
        )

    return dataloader


def collate_langs_dataloader(config, meta_langs, dsn_map):
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    split_names = ["train", "val"]
    dataloaders = dict()

    for split_name in split_names:
        frames = []
        for lang in meta_langs:
            dataset_name = dsn_map.get(lang)
            pkl_path = os.path.join(
                DEST_DATA_PKL_DIR, f"{dataset_name}{lang}_200_{split_name}.pkl"
            )
            cur_data_df = pd.read_pickle(pkl_path, compression=None)
            frames.append(cur_data_df)
        data_df = pd.concat(frames)
        logger.info(f"Total {len(data_df)} samples in {split_name}")
        if config.dataset_type == "bert":
            dataset = HFDataset(
                data_df, config.tokenizer, max_seq_len=config.max_seq_len
            )
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        dataset = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=True if split_name == "train" else False,
        )

        dataloaders[split_name] = dataset
    return dataloaders


if __name__ == "__main__":
    args = parse_helper()
    main(args, cuda=True)

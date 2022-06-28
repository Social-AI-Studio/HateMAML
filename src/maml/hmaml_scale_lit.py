"""
A prototype of domain adaptive meta learning algorithm for multilingual hate detection.
"""
import argparse
import json
import logging
import math
import os
import random

import learn2learn as l2l
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.consts import DEST_DATA_PKL_DIR, RUN_BASE_DIR, SRC_DATA_PKL_DIR
from src.data.datasets import HFDataset
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.model.lightning import LitClassifier
from src.utils import get_single_dataset_from_split, get_collate_langs_dataloader, report_memory


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def manual_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)


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
        "--base_model_path", default=None, type=str, help="Path to fine-tuned base-model, load from checkpoint!"
    )
    parser.add_argument(
        "--dataset_type",
        default="bert",
        type=str,
        help="The input data type. It could take bert, lstm, gpt2 as input.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--shots", default=8, type=int, help="Size of the mini batch")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the mini batch")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
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
    parser.add_argument("--fast_lr", type=float, default=4e-5, help="The inner loop fast adaptation learning rate.")
    parser.add_argument(
        "--overwrite_cache", default=False, action="store_true", help="Overwrite cached results for a run."
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="hmaml_scale",
        help="Provide a MAML training setup. Valid values are 'hmaml_scale'.",
    )
    parser.add_argument(
        "--num_meta_samples", type=int, default=200, help="Number of available samples for meta-training."
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
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    args = parser.parse_args()
    logger.info(args)
    return args


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


def finetune(args, model, train_dataloader, eval_dataloader, test_dataloader_list, meta_langs, device):
    output_dir = os.path.join(RUN_BASE_DIR, "scale", f"{args.exp_setting}", f"seed{args.seed}")
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
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # scheduler related
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_training_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.debug("***** Running training *****")
    logger.debug(f"  Num examples = {len(train_dataloader)*args.batch_size}")
    logger.debug(f"  Num Epochs = {args.num_train_epochs}")
    logger.debug(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.debug(f"  Total optimization steps = {max_training_steps}")

    eval_steps = 50
    nb_train_steps = 0
    min_loss = 1000

    for epoch in tqdm(range(args.num_train_epochs), desc="Iteration"):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Batch"):
            model.zero_grad()
            batch = move_to(batch, device)
            outputs = model(**batch)
            loss, acc = compute_loss_acc(outputs["logits"], batch["labels"], loss_fn)
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_train_loss += loss.item()
            total_train_acc += acc
            nb_train_steps += 1
            if (nb_train_steps + 1) % eval_steps == 0:
                avg_train_loss = total_train_loss / nb_train_steps
                logger.info(f"  Epoch {epoch+1}, step {nb_train_steps+1}, training loss: {avg_train_loss:.3f} \n")

        f1, loss = evaluate(args, model, eval_dataloader, device, type="eval")
        if nb_train_steps > 200 and min_loss > loss:
            min_loss = loss
            logger.info(f"Saving best model checkpoint from epoch {epoch} at {output_dir}")
            model.save_pretrained(output_dir)

    logger.info("============================================================")
    logger.info("============================================================")

    logger.info(f"Loading fine-tuned model from the checkpoint {output_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    summary = {
        "meta_langs": meta_langs,
        "num_meta_iterations": args.num_meta_iterations,
        "base_model_path": args.base_model_path,
        "script_name": os.path.basename(__file__),
        "exp_setting": args.exp_setting,
    }

    for lang_id in meta_langs:
        logger.info(f"Evaluating fine-tuned model performance on test set. Language = {lang_id}")
        if lang_id == "it":
            for lg_id in ["news", "tweets"]:
                eval_result = evaluate(args, model, test_dataloader_list[lg_id], device, type="pred")
                summary[lg_id] = {"f1": eval_result[0], "loss": eval_result[1]}
        else:
            eval_result = evaluate(args, model, test_dataloader_list[lang_id], device, type="pred")
            summary[lang_id] = {"f1": eval_result[0], "loss": eval_result[1]}
    return summary


def main(args, meta_batch_size=None, adaptation_steps=1):
    """
    An implementation of cross-lingual *Model-Agnostic Meta-Learning* algorithm for hate detection
    on low-resouce languages.

    Args:
        meta_lr (float): The learning rate used to update the model.
        fast_lr (float): The learning rate used to update the MAML inner loop.
        adaptation_steps (int); The number of inner loop steps.
        num_iterations (int): The total number of iteration MAML will run (outer loop update).
    """
    f_base_model = "mbert" if "bert" in args.model_name_or_path else "xlmr"
    summary_output_dir = os.path.join(
        "runs/summary", "scale", f_base_model, f"seed{args.seed}", args.exp_setting, str(args.num_meta_samples)
    )
    # etxt = "_" + str(args.num_meta_iterations) if args.exp_setting == "hmaml_scale" else ""
    summary_fname = os.path.join(summary_output_dir, "results.json")
    logger.info(f"Output fname = {summary_fname}")

    meta_langs = args.meta_langs.split(",")
    # meta_langs = ["ar", "da", "gr", "tr", "hi", "de", "es"]
    if os.path.exists(summary_fname) and not args.overwrite_cache:
        return

    manual_seed_all(args.seed)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {args.device} gpu: {args.n_gpu}")

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.base_model_path:
        if "xlm-r" in args.model_name_or_path:
            model = XLMRClassifier()
        elif "bert" in args.model_name_or_path:
            model = MBERTClassifier()
        else:
            raise ValueError(f"Model type {args.model_name_or_path} is unknown.")
        # load the pretrained base model, typicall finetuned on English
        lit_model = LitClassifier(model)
        ckpt = torch.load(os.path.normpath(args.base_model_path), map_location=args.device)
        lit_model.load_state_dict(ckpt["state_dict"])
        model = lit_model.model
        if args.freeze_layers:
            lit_model.set_trainable(True)
            lit_model.set_freeze_layers(args.freeze_layers)
            for name, param in lit_model.model.named_parameters():
                logger.debug("%s - %s", name, ("Unfrozen" if param.requires_grad else "FROZEN"))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    model.to(args.device)

    dsn_map = {
        "en": "hasoc2020",
        "es": "hateval2019",
        "ar": "semeval2020",
        "da": "semeval2020",
        "gr": "semeval2020",
        "tr": "semeval2020",
        "it": "evalita2020",
        "hi": "hasoc2020",
        "de": "hasoc2020",
    }
    if args.exp_setting == "hmaml_scale":
        meta_tasks_list = []
        for lang_id in meta_langs:
            logger.info(f"Loading meta tasks for language = {lang_id}")
            cur_dataloader = get_single_dataset_from_split(
                config=args,
                split_name="train",
                dataset_name=f"{dsn_map.get(lang_id)}{lang_id}",
                lang=lang_id,
                batch_size=args.shots,
            )
            meta_tasks_list.append(cur_dataloader)
        meta_batch_size_list = []
        for l in meta_tasks_list:
            meta_batch_size_list.append(len(l))
        meta_batch_size = sum(meta_batch_size_list)
        logger.info(f"Number of meta tasks {meta_batch_size}")

        # meta_domain_tasks = meta_tasks_list[:]
    elif args.exp_setting == "finetune_collate":
        train_dataloader, val_dataloader = get_collate_langs_dataloader(args, meta_langs, dsn_map)
    else:
        raise ValueError(f"{args.exp_setting} is unknown!")

    test_dataloader_lgs = {}
    for lang_id in meta_langs:
        logger.debug(
            f"**** Zero-shot evaluation on {dsn_map.get(lang_id)} before meta-training. language = {lang_id} ****"
        )
        if lang_id == "it":
            evalita_loaders = get_evalita_loaders(split_name="test", config=args, dataset_name=dsn_map.get(lang_id))
            for kkey in evalita_loaders:
                test_dataloader_lgs[kkey] = evalita_loaders[kkey]
                # evaluate(args, mode, evalita_loaders[kkey], device, type="pred")
        else:
            cur_test_dataloader = get_single_dataset_from_split(
                config=args,
                split_name="test",
                dataset_name=f"{dsn_map.get(lang_id)}{lang_id}",
                lang=lang_id,
                batch_size=args.batch_size,
            )
            test_dataloader_lgs[lang_id] = cur_test_dataloader
            # evaluate(args, model, cur_test_dataloader, device, type="pred")

    if args.exp_setting == "finetune_collate":
        summary = finetune(
            args,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader_lgs,
            meta_langs,
            args.device,
        )
        logger.info(summary)
        os.makedirs(summary_output_dir, exist_ok=True)
        json.dump(summary, open(summary_fname, "w"))
        return

    # MAML training starts here. This step assumes we have a pretrained base model, fine-tuned on English.
    # We will now meta-train the base model with MAML algorithm.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = optim.AdamW(optimizer_grouped_parameters, args.meta_lr, eps=args.adam_epsilon)
    max_training_steps = args.num_meta_iterations * args.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=opt, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_training_steps
    )
    logger.info("*********************************")
    logger.info("*** HATE X MAML training starts now ***")
    logger.info("*********************************")

    loss_fn = torch.nn.CrossEntropyLoss()

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True, allow_unused=True).cuda()
    for iteration in tqdm(range(args.num_meta_iterations), desc="Iteration"):  # outer loop
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_domain_accuracy = 0.0
        tmp_cntr = meta_batch_size_list[:]
        for idx in tqdm(range(meta_batch_size), desc="Meta task"):
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
            task = {k: v.to(args.device) for k, v in task.items()}

            train_query_inp = {
                "input_ids": task["input_ids"][:n_meta_lr],
                "attention_mask": task["attention_mask"][:n_meta_lr],
                "labels": task["labels"][:n_meta_lr],
            }
            train_support_inp = {
                "input_ids": task["input_ids"][n_meta_lr:],
                "attention_mask": task["attention_mask"][n_meta_lr:],
                "labels": task["labels"][n_meta_lr:],
            }
            del task
            learner = maml.clone()
            for _ in range(adaptation_steps):
                outputs = learner(**train_support_inp)
                loss, acc = compute_loss_acc(outputs["logits"], train_support_inp["labels"], loss_fn)

                if args.n_gpu > 1:
                    loss = loss.mean()
                # print("train = ", loss)
                learner.adapt(loss, first_order=True)
                torch.cuda.empty_cache()
                del outputs

            meta_train_error += loss.item()
            meta_train_accuracy += acc
            del acc
            del loss
            del train_support_inp

            outputs = learner(**train_query_inp)
            eval_loss, eval_acc = compute_loss_acc(outputs["logits"], train_query_inp["labels"], loss_fn)

            # print("eval = ", eval_loss)
            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            meta_valid_error += eval_loss
            meta_valid_accuracy += eval_acc
            # eval_loss.backward()
            del outputs
            del eval_acc
            del eval_loss
            del learner
            del train_query_inp
            torch.cuda.empty_cache()
            report_memory(name=f" {idx}")

            if idx > 7:
                break

            # if args.exp_setting in ["hmaml_scale"]:
            #     for k in range(10):
            #         d_idx = random.randint(0, len(meta_domain_tasks) - 1)
            #         if d_idx != choice_idx:
            #             break
            #     b_idx = random.randint(0, len(meta_domain_tasks[d_idx]))

            #     for indx, d_batch in enumerate(meta_domain_tasks[d_idx]):
            #         if indx == b_idx:
            #             d_task = d_batch
            #             break
            #     domain_query_inp = {
            #         "input_ids": d_task["input_ids"][:n_meta_lr],
            #         "attention_mask": d_task["attention_mask"][:n_meta_lr],
            #         "labels": d_task["labels"][:n_meta_lr],
            #     }
            #     domain_query_inp = move_to(domain_query_inp, args.device)
            #     outputs = learner(**domain_query_inp)
            #     d_loss, d_f1 = compute_loss_acc(outputs["logits"], domain_query_inp["labels"], loss_fn)
            #     if args.n_gpu > 1:
            #         d_loss = d_loss.mean()

            #     # total_loss = eval_loss + d_loss
            #     meta_valid_error += d_loss
            #     meta_domain_accuracy += d_f1

            # else:
            #     total_loss = eval_loss
            # total_loss.backward()

        # Print some metrics
        print("\n")
        mt_error = meta_train_error / meta_batch_size
        mt_acc = meta_train_accuracy / meta_batch_size
        mv_error = meta_valid_error / meta_batch_size
        mv_acc = meta_valid_accuracy / meta_batch_size
        md_acc = meta_domain_accuracy / meta_batch_size

        logger.info(
            "  Iteration {} >> Meta Train Error {:.3f}, F1 {:.3f} # Valid Error {:.3f}, F1 {:.3f}, Domain F1 {:.3f}".format(
                iteration, mt_error, mt_acc, mv_error, mv_acc, md_acc
            )
        )

        # Average the accumulated gradients and optimize
        # for p in maml.parameters():
        #     if p.grad is not None:
        #         p.grad.data.mul_(1.0 / meta_batch_size)
        opt.zero_grad()
        mv_error.backward()
        opt.step()
        lr_scheduler.step()
        del meta_train_error
        del meta_train_accuracy
        del meta_valid_error
        del meta_valid_accuracy
        del meta_domain_accuracy
        del mt_error
        del mv_error
        del mt_acc
        del mv_acc
        del md_acc
        torch.cuda.empty_cache()

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
                eval_result = evaluate(args, model, test_dataloader_lgs[lg_id], args.device, type="pred")
                summary[lg_id] = {"f1": eval_result[0], "loss": eval_result[1]}
        else:
            eval_result = evaluate(args, model, test_dataloader_lgs[lang_id], args.device, type="pred")
            summary[lang_id] = {"f1": eval_result[0], "loss": eval_result[1]}

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

        dataset = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=True if split_name != "test" else False,
        )

        dataloaders[split_name] = dataset
    return dataloaders


def get_evalita_loaders(split_name, config, dataset_name):
    evalita_loaders = {}
    topics = ["tweets", "news"]
    for lang_id in topics:
        few_pkl_path = os.path.join(
            DEST_DATA_PKL_DIR,
            f"{dataset_name}_{lang_id}_{split_name}.pkl",
        )
        data_df = pd.read_pickle(few_pkl_path, compression=None)
        logger.debug(f"picking {data_df.shape[0]} rows from `{few_pkl_path}` as few samples")

        if config.dataset_type == "bert":
            dataset = HFDataset(data_df, config.tokenizer, max_seq_len=config.max_seq_len)
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        evalita_loaders[lang_id] = dataloader
    return evalita_loaders


if __name__ == "__main__":
    args = parse_helper()
    main(args)

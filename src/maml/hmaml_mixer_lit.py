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
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
from src.data.datasets import HFDataset
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.model.lightning import LitClassifier

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_helper():
    parser = argparse.ArgumentParser()
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
        default="runs/baselines/founta/en/full/Mbert1.ckpt",
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
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--shots", default=8, type=int, help="Size of the mini batch")
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
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="The maximum sequence length of the inputs.",
    )
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
        "--load_saved_base_model",
        default=False,
        action="store_true",
        help="Fine-tune base-model loading from a given checkpoint.",
    )
    parser.add_argument(
        "--overwrite_cache", default=False, action="store_true", help="Overwrite cached results for a run."
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="hmaml",
        help="Provide a choice for MAML training. Valid values are 'xmaml', 'metra', 'hmaml', 'hmaml-refine'.",
    )
    parser.add_argument(
        "--metatune_type",
        type=str,
        default=None,
        help="Meta-adaptation flag, either `zeroshot` or `fewshot`. Used in target language MAML training on only n samples where `n` = 500.",
    )
    parser.add_argument(
        "--fft",
        action="store_true",
        help="Further fine-tune on target training set, initialized from the zeroshot metatuned checkpoint.",
    )
    parser.add_argument(
        "--refine_threshold", type=float, default=0.90, help="The threshold value for filtering silver labels."
    )
    parser.add_argument(
        "--num_meta_samples", type=int, default=None, help="Number of available samples for meta-training."
    )
    parser.add_argument(
        "--freeze_layers",
        type=str,
        default=None,
        help='Set freeze layers. `freeze_layers` can only be in ["embeddings","top3","top6"]',
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    args = parser.parse_args()
    logger.info(args)
    return args


def manual_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)


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
    run_dir = os.path.join(RUN_BASE_DIR, "fft", f"{args.dataset_name}{args.target_lang}")
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, args.model_name_or_path)
    logger.debug(f"Saving fine-tuned checkpoint at {output_dir}")
    torch.save(model.state_dict(), output_dir)


def evaluate(args, model, eval_dataloader, device, type="test"):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    pred_label = []
    target_label = []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = move_to(batch, device)
            outputs = model(batch)
            logits = outputs["logits"]
            pred_label.append(logits)
            target_label.append(batch["labels"])

    pred_label = torch.cat(pred_label)
    target_label = torch.cat(target_label)
    loss, f1 = compute_loss_acc(pred_label, target_label, loss_fn)
    loss = loss.item()
    log_str = "prediction" if type == "pred" else "evaluation"
    if type == "pred":
        logger.info("*** Running {} **".format(log_str))
        logger.info(f"  Num examples = {len(eval_dataloader)*args.batch_size}")
        logger.info(f"  Loss = {loss:.3f}")
        logger.info(f"  F1 = {f1:.3f}")

    return f1, loss


def finetune(args, model, train_dataloader, eval_dataloader, test_dataloader, device):
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
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_training_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.debug("***** Running training *****")
    logger.debug(f"  Num examples = {len(train_dataloader)*args.batch_size}")
    logger.debug(f"  Num Epochs = {args.num_train_epochs}")
    logger.debug(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.debug(f"  Total optimization steps = {max_training_steps}")

    nb_train_steps = 0
    min_macro_f1 = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            batch = move_to(batch, device)
            outputs = model(batch)
            loss, acc = compute_loss_acc(outputs["logits"], batch["labels"], loss_fn)
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            total_train_acc += acc
            nb_train_steps += 1

            if (nb_train_steps + 1) % args.eval_steps == 0:
                avg_train_loss = total_train_loss / nb_train_steps
                logger.debug(f"  epoch = {epoch+1}, step = {nb_train_steps+1}, loss = {avg_train_loss:.3f} \n")

        f1, _ = evaluate(args, model, eval_dataloader, device, type="eval")
        if min_macro_f1 < f1:
            min_macro_f1 = f1
            checkpoint_ft_model(args, model)

    logger.info("============================================================")
    logger.info(f"Evaluating fine-tuned model performance on test set. Training lang = {args.target_lang}")
    logger.info("============================================================")

    path_to_model = os.path.join(RUN_BASE_DIR, "fft", f"{args.dataset_name}{args.target_lang}", args.model_name_or_path)
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
    f1, loss = evaluate(args, model, test_dataloader, device, type="pred")

    result = {"examples": len(train_dataloader) * args.batch_size, "f1": f1, "loss": loss}
    return result


def main(args, meta_batch_size=None, adaptation_steps=1):
    """
    An implementation of two-step *Model-Agnostic Meta-Learning* algorithm for hate detection
    on low-resouce languages.

    Args:
        meta_lr (float): The learning rate used to update the model.
        fast_lr (float): The learning rate used to update the MAML inner loop.
        adaptation_steps (int); The number of inner loop steps.
        num_iterations (int): The total number of iteration MAML will run (outer loop update).
    """

    summary_output_dir = os.path.join("runs/summary", args.dataset_name, os.path.basename(__file__)[:-3])
    aux_la = "_" + args.aux_lang if args.aux_lang else ""
    few_ft = "_" + args.metatune_type
    f_base_model = args.base_model_path[-11:-6] if "bert" in args.base_model_path else args.base_model_path[-10:-6]
    f_samples = "_" + str(args.num_meta_samples) if args.num_meta_samples else ""
    summary_fname = os.path.join(
        summary_output_dir,
        f"{f_base_model}{args.seed}_{args.exp_setting}_{args.num_meta_iterations}_{args.shots//2}{few_ft}{f_samples}{aux_la}_{args.target_lang}.json",
    )
    logger.info(f"Output fname = {summary_fname}")

    if os.path.exists(summary_fname) and not args.overwrite_cache:
        return

    manual_seed_all(args.seed)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"device: {args.device} gpu: {args.n_gpu}")

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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

    if args.freeze_layers:
        lit_model.set_trainable(True)
        lit_model.set_freeze_layers(args.freeze_layers)
        for name, param in lit_model.model.named_parameters():
            logger.debug("%s - %s", name, ("Unfrozen" if param.requires_grad else "FROZEN"))

    model = lit_model.model
    model.to(args.device)

    args.src_dataset_name = "founta" if args.dataset_name == "semeval2020" else args.dataset_name

    # test set of target language, for ultimate evaluation
    testlg_dataloader = get_single_dataset_from_split(
        config=args,
        split_name="test",
        dataset_name=args.dataset_name + args.target_lang,
        lang=args.target_lang,
        batch_size=args.batch_size,
    )
    logger.info(
        f"**** Zero-shot evaluation on {args.dataset_name} before meta-tuning >>> language = {args.target_lang} ****"
    )
    evaluate(args, model, testlg_dataloader, args.device, type="pred")

    # MAML training starts here. This step assumes we have a pretrained base model, fine-tuned on English.
    # We will now meta-train the base model with MAML algorithm.

    meta_train_tasks, meta_domain_tasks, trlg_fft_dataloader = prepare_meta_tuning_tasks(args, model)
    meta_batch_size = len(meta_train_tasks)
    logger.info(f"Number of meta-training tasks {meta_batch_size}")
    if args.exp_setting == "hmaml":
        domains = list(meta_domain_tasks.keys())

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    opt = optim.Adam(maml.parameters(), lr=args.meta_lr, eps=args.adam_epsilon)
    # max_training_steps = args.num_meta_iterations * args.gradient_accumulation_steps
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=opt, num_warmup_steps=0, num_training_steps=max_training_steps
    # )

    logger.info("*************************************")
    logger.info("*** HATE X META TRAINING STARTS NOW ***")
    logger.info("*************************************")

    loss_fn = torch.nn.CrossEntropyLoss()

    for iteration in tqdm(range(args.num_meta_iterations), desc="Iteration"):  # outer loop
        opt.zero_grad()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_domain_accuracy = 0.0

        for task in tqdm(meta_train_tasks, total=meta_batch_size, desc="Meta task"):
            n_meta_lr = args.shots // 2

            if isinstance(meta_train_tasks, list):
                train_query_inp = task["query"]
                train_support_inp = task["support"]
            else:
                tname = task["task_name"]
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

            train_query_inp = move_to(train_query_inp, args.device)
            train_support_inp = move_to(train_support_inp, args.device)

            learner = maml.clone()
            # report_memory(name = f"maml{idx}")

            for _ in range(adaptation_steps):
                outputs = learner(train_support_inp)
                loss, acc = compute_loss_acc(outputs["logits"], train_support_inp["labels"], loss_fn)

                if args.n_gpu > 1:
                    loss = loss.mean()
                learner.adapt(loss, allow_nograd=True, allow_unused=True)
                meta_train_error += loss.item()
                meta_train_accuracy += acc
            outputs = learner(train_query_inp)
            eval_loss, eval_acc = compute_loss_acc(outputs["logits"], train_query_inp["labels"], loss_fn)

            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            meta_valid_error += eval_loss
            meta_valid_accuracy += eval_acc

            if args.exp_setting == "hmaml":
                if tname == domains[0]:
                    dname = domains[1]
                else:
                    dname = domains[0]
                choice_idx = random.randint(0, len(meta_domain_tasks[dname]) - 1)
                d_task = meta_domain_tasks[dname][choice_idx]
                domain_query_inp = {
                    "input_ids": d_task["input_ids"][:n_meta_lr],
                    "attention_mask": d_task["attention_mask"][:n_meta_lr],
                    "labels": d_task["labels"][:n_meta_lr],
                }
                domain_query_inp = move_to(domain_query_inp, args.device)
                outputs = learner(domain_query_inp)
                d_loss, d_f1 = compute_loss_acc(outputs["logits"], domain_query_inp["labels"], loss_fn)
                if args.n_gpu > 1:
                    d_loss = d_loss.mean()

                meta_valid_error += d_loss
                total_loss = eval_loss + d_loss
                meta_domain_accuracy += d_f1

            else:
                total_loss = eval_loss

            total_loss.backward()

        # Print some metrics
        print("\n")
        mt_error = meta_train_error / (meta_batch_size * adaptation_steps)
        mt_acc = meta_train_accuracy / (meta_batch_size * adaptation_steps)
        mv_error = meta_valid_error / meta_batch_size
        mv_acc = meta_valid_accuracy / meta_batch_size
        md_acc = meta_domain_accuracy / meta_batch_size

        logger.info(
            "  Iteration {} >> Meta-train error {:.3f}, F1 {:.3f} # Validation error {:.3f}, F1 {:.3f}, Domain F1 {:.3f}".format(
                iteration, mt_error, mt_acc, mv_error, mv_acc, md_acc
            )
        )

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)
        # mv_error.backward()
        opt.step()
        # lr_scheduler.step()

        if mv_error < 0.01:
            break

        # TODO: requires fixing
        if args.exp_setting == "hmaml-refine":
            tgt_lg_dev_dataloader = get_single_dataset_from_split(
                config=args,
                split_name="val",
                dataset_name=f"{args.dataset_name}{args.target_lang}",
                lang=args.target_lang,
            )
            silver_dataset = get_silver_dataset_for_meta_refine(args, model, tgt_lg_dev_dataloader, args.device)
            dataloaders = {}
            dataloaders[args.source_lang] = DataLoaderWithTaskname(
                task_name=args.source_lang,
                data_loader=get_single_dataset_from_split(
                    config=args,
                    split_name="val",
                    dataset_name=f"{args.src_dataset_name}{args.source_lang}",
                    lang=args.source_lang,
                    batch_size=args.shots,
                ),
            )
            dataloaders["silver"] = DataLoaderWithTaskname(
                task_name="silver",
                data_loader=DataLoader(
                    silver_dataset, batch_size=args.shots, num_workers=args.num_workers, shuffle=True, drop_last=True
                ),
            )
            meta_train_tasks = MultitaskDataloader(dataloaders)
            logger.info(f"Number of meta tasks {len(meta_train_tasks)}")

    # Zero-shot, Few-shot or Full-tuned evaluation? You can now take the zero-shot meta-trained model and fine-tune it on
    # available low-resource few-shot training samples. If we don't fine-tune further, it can be considered as zero-shot model.
    summary = {
        "aux_lang": args.aux_lang,
        "target_lang": args.target_lang,
        "num_meta_iterations": args.num_meta_iterations,
        "base_model_path": args.base_model_path,
        "script_name": os.path.basename(__file__),
        "exp_setting": args.exp_setting,
        "fft": args.fft,
    }

    logger.info(
        f"**** evaluation = {args.metatune_type} dataset_name =  {args.dataset_name} exp = {args.exp_setting} >>> language = {args.target_lang} ****"
    )
    metatune_result = evaluate(args, model, testlg_dataloader, args.device, type="pred")
    summary[str(args.exp_setting)] = {"f1": metatune_result[0], "loss": metatune_result[1]}

    # **** THE FURTHER FINE-TUNING STEP ****

    if args.fft:
        fft_result = finetune(
            args,
            model,
            trlg_fft_dataloader,
            trlg_fft_dataloader,
            testlg_dataloader,
            args.device,
        )
        summary["fft"] = fft_result

    os.makedirs(summary_output_dir, exist_ok=True)
    json.dump(summary, open(summary_fname, "w"))


def prepare_meta_tuning_tasks(args, model=None):
    # source langage dev set required for hmaml or xmetra
    # select auxiliary language dev set if zeroshot, else target language dev set
    # choose meta domain task if only hmaml
    tune_dataset_langs = {args.source_lang: f"{args.src_dataset_name}{args.source_lang}"}
    if args.metatune_type in ["zeroshot", "refine"]:
        other_lang = args.aux_lang
    elif args.metatune_type == "fewshot":
        other_lang = args.target_lang
    else:
        raise ValueError(f"Invalid value for `args.metatune_type`, found = {args.metatune_type}")

    tune_dataset_langs[other_lang] = f"{args.dataset_name}{other_lang}"

    if args.exp_setting == "hmaml":
        if args.metatune_type != "refine":
            dataloaders = {}
            for lang, dataset_name in tune_dataset_langs.items():
                dataloader = DataLoaderWithTaskname(
                    task_name=lang,
                    data_loader=get_single_dataset_from_split(
                        config=args,
                        split_name="val",
                        dataset_name=dataset_name,
                        lang=args.source_lang,
                        batch_size=args.shots,
                    ),
                )
                dataloaders[lang] = dataloader
            meta_train_tasks = MultitaskDataloader(dataloaders)
        else:
            tgt_lg_dev_dataloader = get_single_dataset_from_split(
                config=args,
                split_name="val",
                dataset_name=f"{args.dataset_name}{args.target_lang}",
                lang=args.target_lang,
            )
            silver_dataset = get_silver_dataset_for_meta_refine(args, model, tgt_lg_dev_dataloader, args.device)
            dataloaders = {}
            dataloaders[args.source_lang] = DataLoaderWithTaskname(
                task_name=args.source_lang,
                data_loader=get_single_dataset_from_split(
                    config=args,
                    split_name="val",
                    dataset_name=tune_dataset_langs[args.source_lang],
                    lang=args.source_lang,
                    batch_size=args.shots,
                ),
            )
            dataloaders["silver"] = DataLoaderWithTaskname(
                task_name="silver",
                data_loader=DataLoader(
                    silver_dataset, batch_size=args.shots, num_workers=args.num_workers, shuffle=True, drop_last=True
                ),
            )
            meta_train_tasks = MultitaskDataloader(dataloaders)
    elif args.exp_setting == "xmetra":
        meta_train_tasks = []
        support_dataloader = get_single_dataset_from_split(
            config=args,
            split_name="val",
            dataset_name=tune_dataset_langs[args.source_lang],
            lang=args.source_lang,
            batch_size=args.shots // 2,
        )

        query_dataloader = get_single_dataset_from_split(
            config=args,
            split_name="val",
            dataset_name=tune_dataset_langs[other_lang],
            lang=args.source_lang,
            batch_size=args.shots // 2,
        )

        spt_tasks = [spt for spt in support_dataloader]
        qry_tasks = [qry for qry in query_dataloader]

        for i in range(len(qry_tasks)):
            meta_train_tasks.append({"support": spt_tasks[i], "query": qry_tasks[i]})

    elif args.exp_setting == "xmaml":
        dataloaders = {}
        dataloaders[other_lang] = DataLoaderWithTaskname(
            task_name=other_lang,
            data_loader=get_single_dataset_from_split(
                config=args,
                split_name="val",
                dataset_name=tune_dataset_langs[other_lang],
                lang=other_lang,
                batch_size=args.shots,
            ),
        )
        meta_train_tasks = MultitaskDataloader(dataloaders)
    else:
        raise ValueError(f"{args.exp_setting} is unknown!")

    # domain task only required for hmaml
    meta_domain_tasks = None
    if args.exp_setting == "hmaml":
        src_tasks = get_single_dataset_from_split(
            config=args,
            split_name="val",
            dataset_name=tune_dataset_langs[args.source_lang],
            lang=other_lang,
            batch_size=args.shots,
        )
        oth_tasks = get_single_dataset_from_split(
            config=args,
            split_name="val",
            dataset_name=tune_dataset_langs[other_lang],
            lang=other_lang,
            batch_size=args.shots,
        )
        src_tasks = [task for task in src_tasks]
        oth_tasks = [task for task in oth_tasks]

        meta_domain_tasks = {args.source_lang: src_tasks, other_lang: oth_tasks}

    # futher finetune on zeroshot metatuned checkpoint
    trlg_fft_dataloader = None
    if args.fft and args.metatune_type == "zeroshot":
        trlg_fft_dataloader = get_single_dataset_from_split(
            config=args,
            split_name="val",
            dataset_name=f"{args.dataset_name}{args.target_lang}",
            lang=args.target_lang,
            to_shuffle=True,
            batch_size=args.batch_size,
        )

    return meta_train_tasks, meta_domain_tasks, trlg_fft_dataloader


def get_silver_dataset_for_meta_refine(args, model, dataloader, device):
    logger.info(f"Generating silver labels for target language {args.target_lang}")

    model.eval()

    silver_dataset = None
    threshold = args.refine_threshold
    for idx, batch in enumerate(dataloader):
        bindices = []
        batch["labels"] = batch.pop("label")
        with torch.no_grad():
            batch = move_to(batch, device)
            outputs = model(batch)

            pred_label = outputs["logits"]

            probabilities = F.softmax(pred_label, dim=-1)

            for values in probabilities:
                if values[0] > threshold or values[1] > threshold:
                    bindices.append(True)
                else:
                    bindices.append(False)

            cols = ["input_ids", "attention_mask", "labels"]
            silver_batch = dict()
            for col in cols:
                silver_batch[col] = batch[col][bindices].cpu().numpy()

            if silver_dataset is None:
                silver_dataset = silver_batch
            else:
                for key, value in silver_batch.items():
                    silver_dataset[key] = np.concatenate((silver_dataset[key], value), axis=0)

    unique, counts = np.unique(silver_dataset["labels"], return_counts=True)
    lower_bnd = min(min(counts), 250)

    all_items = []
    for idx in range(len(silver_dataset["labels"])):
        item = {}
        for key in silver_dataset.keys():
            item[key] = silver_dataset[key][idx]
        all_items.append(item)
    random.shuffle(all_items)

    filtered_items = []
    key_cnt = {k: k for k in unique}
    for item in all_items:
        key_cnt[item["labels"]] += 1
        if key_cnt[item["labels"]] <= lower_bnd:
            filtered_items.append(item)

    silver_tdataset = SilverDataset(filtered_items)
    logger.info(f"Loading refined silver dataset with {len(filtered_items)} training samples.")
    return silver_tdataset


class SilverDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.data_dict[idx].items()}
        item["label"] = item.pop("labels")
        return item

    def __len__(self):
        return len(self.data_dict)


def get_single_dataset_from_split(config, split_name, dataset_name=None, lang=None, to_shuffle=False, batch_size=None):
    if split_name != "test" and config.num_meta_samples:
        data_pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}_{config.num_meta_samples}_{split_name}.pkl")
    else:
        data_pkl_path = os.path.join(DEST_DATA_PKL_DIR, f"{dataset_name}_{split_name}.pkl")

    data_df = pd.read_pickle(data_pkl_path, compression=None)
    data_df = data_df.sample(frac=1)
    logger.info(f"picking {data_df.shape[0]} rows from `{data_pkl_path}`")

    if config.dataset_type == "bert":
        dataset = HFDataset(data_df, config.tokenizer, max_seq_len=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        shuffle=to_shuffle,  # default set to False for meta-training as fixed batch represent an uniqe task
        drop_last=False if split_name == "test" else True,
    )

    return dataloader


if __name__ == "__main__":
    args = parse_helper()
    main(args)

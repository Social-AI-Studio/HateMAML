
"""
A prototype of domain adaptive meta learning algorithm for multilingual hate detection.
"""
import argparse
import json
import logging
import math
import os
import random
import copy

import higher
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import optim
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

from mtl_datasets import DataLoaderWithTaskname, MultitaskDataloader
from src.data.consts import RUN_BASE_DIR
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.utils import get_single_dataloader_from_split, SilverDataset, get_dataloaders_from_split

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
        "--base_model_path", default=None, type=str, help="Path to fine-tuned base-model, load from checkpoint!"
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
    parser.add_argument("--num_meta_iterations", type=int, default=None, help="Number of outer loop iteratins.")
    parser.add_argument("--meta_lr", type=float, default=2e-5, help="The outer loop meta update learning rate.")
    parser.add_argument("--fast_lr", type=float, default=4e-5, help="The inner loop fast adaptation learning rate.")
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
    parser.add_argument(
        "--eval_steps", type=int, default=50, help="Number of steps interval for model evaluation."
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


def finetune(args, nmodel, train_dataloader, eval_dataloader, test_dataloader, device):
    model = copy.deepcopy(nmodel)
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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)*args.batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.debug(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.debug(f"  Total optimization steps = {max_training_steps}")

    path_to_model = os.path.join(RUN_BASE_DIR, "fft", f"{args.dataset_name}{args.target_lang}", args.model_name_or_path)
    nb_train_steps = 0
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_train_epochs)):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            batch = move_to(batch, device)
            outputs = model(**batch)
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

        f1, loss = evaluate(args, model, eval_dataloader, device, type="eval")
        if best_loss > loss:
            best_loss = loss
            model.save_pretrained(path_to_model)

    logger.info("============================================================")
    logger.info(f"Evaluating fine-tuned model performance on test set. Training lang = {args.target_lang}")
    logger.info("============================================================")

    logger.info(f"Loading fine-tuned model from the checkpoint {path_to_model}")

    model = AutoModelForSequenceClassification.from_pretrained(path_to_model)
    model.to(device)
    f1, loss = evaluate(args, model, test_dataloader, device, type="pred")

    result = {"examples": len(train_dataloader) * args.batch_size, "f1": f1, "loss": loss}
    return result


def main(args, meta_batch_size=None, adaptation_steps=5):
    """
    An implementation of cross-lingual *Model-Agnostic Meta-Learning* algorithm for hate detection
    on low-resouce languages.

    Args:
        meta_lr (float): The learning rate used to update the model.
        fast_lr (float): The learning rate used to update the MAML inner loop.
        adaptation_steps (int); The number of inner loop steps.
        num_iterations (int): The total number of iteration MAML will run (outer loop update).
    """

    summary_output_dir = os.path.join("runs/summary", args.dataset_name, os.path.basename(__file__)[:-3])
    aux_la = "_" + args.aux_lang if args.aux_lang else ""
    few_ft = "_" + args.metatune_type if args.metatune_type else ""
    f_base_model = "mbert" if "bert-base" in args.model_name_or_path else "xlmr"
    summary_fname = os.path.join(
        summary_output_dir,
        f"{f_base_model}{args.seed}_{args.exp_setting}_{args.shots//2}{few_ft}{aux_la}_{args.target_lang}.json",
    )
    logger.info(f"Output fname = {summary_fname}")

    if os.path.exists(summary_fname) and not args.overwrite_cache:
        return

    manual_seed_all(args.seed)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {args.device} gpu: {args.n_gpu}")

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.base_model_path:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model_path)

    model.to(args.device)

    args.src_dataset_name = "founta" if args.dataset_name == "semeval2020" else args.dataset_name

    # test set of target language, for ultimate evaluation
    logger.info("Generating test set")
    testlg_dataloader = get_single_dataloader_from_split(
        config=args,
        split_name="test",
        dataset_name=args.dataset_name + args.target_lang,
        lang=args.target_lang,
        batch_size=args.batch_size,
    )

    summary = {
        "aux_lang": args.aux_lang,
        "target_lang": args.target_lang,
        "num_meta_iterations": args.num_meta_iterations,
        "base_model_path": args.base_model_path,
        "script_name": os.path.basename(__file__),
        "exp_setting": args.exp_setting,
        "result": [],
    }

    logger.info(
        f"**** Zero-shot evaluation on {args.dataset_name} before meta-tuning >>> language = {args.target_lang} ****"
    )
    zeroshot_result = evaluate(args, model, testlg_dataloader, args.device, type="pred")
    summary["zero"] = {"f1": zeroshot_result[0], "loss": zeroshot_result[1]}

    # MAML training starts here. This step assumes we have a pretrained base model, fine-tuned on English.
    # We will now meta-train the base model with MAML algorithm.

    
    train_dataloader = get_single_dataloader_from_split(
        config=args,
        split_name="train",
        dataset_name=f"{args.dataset_name}{args.aux_lang}_{args.num_meta_samples}",
        lang=args.aux_lang,
        to_shuffle=True,
        batch_size=args.shots,
    )
    meta_batch_size = len(train_dataloader)
    logger.info(f"Number of meta-training tasks {meta_batch_size}")

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
    opt = optim.Adam(optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon)
    max_training_steps = args.num_meta_iterations * args.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=opt, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_training_steps
    )

    logger.info("*************************************")
    logger.info("*** HATE X META TRAINING STARTS NOW ***")
    logger.info("*************************************")

    for iteration in tqdm(range(args.num_meta_iterations), desc='Iteration'):  # outer loop
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.fast_lr)
        meta_train_error = 0.0
        meta_valid_error = 0.0

        for task in tqdm(train_dataloader, desc="Meta task"):
            n_meta_lr = args.shots // 2
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

            train_query_inp = {k: v.to(args.device) for k, v in train_query_inp.items()}
            train_support_inp = {k: v.to(args.device) for k, v in train_support_inp.items()}

            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):
                fast_model.train()
                loss = fast_model(**train_support_inp)["loss"]
                if args.n_gpu > 1:
                    loss = loss.mean()
                diffopt.step(loss)
                meta_train_error += loss.item()

            eval_loss = fast_model(**train_query_inp)["loss"]

            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            eval_loss.backward()
            meta_valid_error += eval_loss

        # Print some metrics
        print("\n")
        mt_error = meta_train_error / meta_batch_size
        mv_error = meta_valid_error / meta_batch_size

        logger.info(
            "  Iteration {} >> Meta-train error {:.3f} # Validation error {:.3f}".format(
                iteration, mt_error, mv_error
            )
        )

        opt.step()
        model.zero_grad()
        lr_scheduler.step()

        if iteration > 25 and (iteration+1) % 5 == 0:
            info = {"iteration": str(iteration)}
            metatune_result = evaluate(args, model, testlg_dataloader, args.device, type="pred")
            info["meta"] = {"f1": metatune_result[0], "loss": metatune_result[1]}
            summary['result'].append(info)


    # Zero-shot, Few-shot or Full-tuned evaluation? You can now take the zero-shot meta-trained model and fine-tune it on
    # available low-resource few-shot training samples. If we don't fine-tune further, it can be considered as zero-shot model.
    logger.info(
        f"**** evaluation = {args.metatune_type} dataset_name =  {args.dataset_name} exp = {args.exp_setting} >>> language = {args.target_lang} ****"
    )

    os.makedirs(summary_output_dir, exist_ok=True)
    json.dump(summary, open(summary_fname, "w"))

if __name__ == "__main__":
    args = parse_helper()
    main(args)

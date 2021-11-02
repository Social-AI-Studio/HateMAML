"""
A prototype of two step meta learning algorithm for multilingual hate detection.
"""
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
from src.model.classifiers import MBERTClassifier
from src.model.lightning import LitClassifier
import torch
import torch.nn.functional as F
import learn2learn as l2l

from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, \
    AutoTokenizer
from src.data.datasets import HFDataset
from transformers import logging as tflog
tflog.set_verbosity_error()

run_suffix = datetime.datetime.now().strftime("%Y_%m_%d_%H")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                              '%m-%d-%Y %H:%M:%S')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f"runs/logs/logs_{run_suffix}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logging.basicConfig(
                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def parse_helper():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="data/processed/", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: `mbert`, `xlm-r`")
    parser.add_argument("--base_model_path", default="runs/tanmoy/Mbert2.ckpt", type=str,
                        help="Path to fine-tunes base-model, load from checkpoint!")
    parser.add_argument("--dataset_type", default="bert", type=str,
                        help="The input data type. It could take bert, lstm, gpt2 as input.")
    parser.add_argument("--dataset_name", default="semeval2020", type=str,
                        help="The name of the task to train selected in the list: [semeval, hateval, hasoc]",
                        )
    # other parameters
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--shots",
                        default=8,
                        type=int,
                        help="Size of the mini batch")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Size of the mini batch")
    parser.add_argument("--source_lang", type=str, default=None, required=True,
                        help="Languages to use for inital fine-tuning that produces the `base-model`.")
    parser.add_argument("--aux_lang", type=str, default=None,
                        help="Auxiliary languages to use for meta-training.")
    parser.add_argument("--target_lang", type=str, default=None, required=True,
                        help="After finishing meta-training, meta-tuned model evaluation on the target language.")
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="The maximum sequence length of the inputs.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="The number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for tokenization.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="Number of gpus to use.")
    parser.add_argument("--device_id", type=int, default=3,
                        help="Gpu id to use.")    
    parser.add_argument("--num_meta_iterations", type=int, default=10,
                        help="Number of outer loop iteratins.")
    parser.add_argument("--meta_lr", type=float, default=2e-5,
                        help="The outer loop meta update learning rate.")
    parser.add_argument("--fast_lr", type=float, default=4e-5,
                        help="The inner loop fast adaptation learning rate.")
    parser.add_argument("--load_saved_base_model", default=False, action='store_true',
                        help="Fine-tune base-model loading from a given checkpoint.")  
    parser.add_argument("--overwrite_cache", default=False, action='store_true',
                        help="Overwrite cached results for a run.")  

    parser.add_argument("--exp_setting", type=str, default='maml-step1',
                        help="Provide a MAML training setup. Valid values are 'xmaml', 'maml-step1', 'maml-step1_2', 'maml-refine'.")             
    parser.add_argument("--finetune_fewshot", type=str, default=None,
                        help="Meta-adaptation (step 2) flag. Used in target language MAML training on only n samples where `n` = 200.")  
    parser.add_argument("--refine_threshold", type=float, default=0.95,
                        help="The threshold value for filtering silver labels.")             

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

    f1 = f1_score(preds, labels, average='macro')
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
            batch = move_to(batch ,device)
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

    nb_train_steps = 0
    min_macro_f1 = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            model.zero_grad()
            batch["labels"] = batch.pop("label")
            batch  = move_to(batch, device)
            outputs = model(batch)
            loss, acc = compute_loss_acc(outputs["logits"], batch["labels"], loss_fn)
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            opt.step()
            total_train_loss += loss.item()
            total_train_acc += acc
            if (nb_train_steps + 1) % 5 == 0:
                avg_train_loss = total_train_loss / nb_train_steps
                logger.debug(f"  Epoch {epoch+1}, step {nb_train_steps+1}, training loss: {avg_train_loss:.3f} \n")
                f1, _ = evaluate(args, model, eval_dataloader, device, type="eval")
                if min_macro_f1 < f1:
                    min_macro_f1 = f1
                    checkpoint_ft_model(args, model)

            nb_train_steps += 1

    logger.info("============================================================")
    logger.info(f"Evaluating fine-tuned model performance on test set. Training lang = {args.target_lang}")
    logger.info("============================================================")


 
    path_to_model = os.path.join(RUN_BASE_DIR, "ft", f"{args.dataset_name}{args.target_lang}", args.model_name_or_path)
    logger.info(f"Loading fine-tuned model from the checkpoint {path_to_model}")
    
    model = MBERTClassifier()
    ckpt = torch.load(os.path.normpath(path_to_model))
    model.load_state_dict(ckpt)
    model.to(device)    
    f1, loss = evaluate(args, model, test_dataloader, device, type="pred")

    result = {"examples": len(train_dataloader)*args.batch_size,  "f1": f1, "loss": loss}
    return result


def main(args,
         meta_batch_size=None,
         adaptation_steps=1,
         cuda=False,
         seed=42):
    """
    An implementation of two-step *Model-Agnostic Meta-Learning* algorithm for hate detection 
    on low-resouce languages.

    Args:
        meta_lr (float): The learning rate used to update the model.
        fast_lr (float): The learning rate used to update the MAML inner loop.
        adaptation_steps (int); The number of inner loop steps.
        num_iterations (int): The total number of iteration MAML will run (outer loop update).
    """

    summary_output_dir= os.path.join("runs/summary", os.path.basename(__file__)[:-3])
    aux_la = "_" + args.aux_lang if args.aux_lang else ""
    summary_fname = os.path.join(summary_output_dir, f"{args.base_model_path[-11:-5]}_{args.exp_setting}_{args.num_meta_iterations}_{args.shots//2}{aux_la}_{args.target_lang}.json")
    if os.path.exists(summary_fname) and not args.overwrite_cache:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device(f'cuda:{args.device_id}')
        args.n_gpu = 1
    logger.debug(f"device : {device}")
    logger.debug(f"gpu : {args.n_gpu}")
    
    model = MBERTClassifier()
    lit_model = LitClassifier(model)
    ckpt = torch.load(os.path.normpath(args.base_model_path))
    lit_model.load_state_dict(ckpt['state_dict'])
    model = lit_model.model
    model.to(device)

    target_lang_dataloaders = get_split_dataloaders(args, dataset_name=args.dataset_name, lang=args.target_lang)

    if args.exp_setting == "hmaml-zeroshot":
        meta_tasks_list = [get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"founta{args.source_lang}", lang=args.source_lang
        ),
        get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"{args.dataset_name}{args.aux_lang}", lang=args.aux_lang
        )]
        meta_batch_size_list = []
        for l in meta_tasks_list:
            meta_batch_size_list.append(len(l))
        meta_batch_size = sum(meta_batch_size_list)
        logger.info(f"Number of meta tasks {meta_batch_size}")

        meta_domain_tasks = get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"{args.dataset_name}{args.aux_lang}", lang=args.aux_lang
        )

        if args.finetune_fewshot == "few":
            train_target_ft_dataloader = get_dataloader(
                split_name="train", config=args, train_few_dataset_name= f"{args.dataset_name}{args.target_lang}", lang=args.target_lang, train = "standard",
            )
            eval_target_ftdataloader = get_dataloader(
                split_name="val", config=args, train_few_dataset_name= f"{args.dataset_name}{args.target_lang}", lang=args.target_lang, train = "standard",
            )

    elif args.exp_setting == "hmaml-fewshot":
        meta_tasks_list = [get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"founta{args.source_lang}", lang=args.source_lang
        ),
        get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"{args.dataset_name}{args.target_lang}", lang=args.target_lang
        )]
        meta_batch_size_list = []
        for l in meta_tasks_list:
            meta_batch_size_list.append(len(l))
        meta_batch_size = sum(meta_batch_size_list)
        logger.info(f"Number of meta tasks {meta_batch_size}")

        meta_domain_tasks = get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"{args.dataset_name}{args.target_lang}", lang=args.target_lang
        )
    elif args.exp_setting == "hmaml-zero-refine":
        silver_dataset = get_silver_dataset_for_meta_refine(args, model, target_lang_dataloaders['train'], device)
        meta_tasks_list = [get_dataloader(
            split_name="train", config=args, train_few_dataset_name= f"founta{args.source_lang}", lang=args.source_lang
        ),
        torch.utils.data.DataLoader(
            silver_dataset, batch_size=args.shots, num_workers=args.num_workers, shuffle=True, drop_last = True,
        )]

        meta_batch_size_list = []
        for l in meta_tasks_list:
            meta_batch_size_list.append(len(l))
        meta_batch_size = sum(meta_batch_size_list)
        logger.info(f"Number of meta tasks {meta_batch_size}")
    else:
       raise ValueError(f"{args.exp_setting} is unknown!")
    
    logger.info(f"**** Zero-shot evaluation on {args.dataset_name} before MAML >>> Language = {args.target_lang} ****")
    evaluate(args, model, target_lang_dataloaders['test'], device, type="pred")

    # MAML training starts here. This step assumes we have a pretrained base model, fine-tuned on English. 
    # We will now meta-train the base model with MAML algorithm.

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    opt = optim.AdamW(maml.parameters(), args.meta_lr)

    logger.info("*********************************")
    logger.info("*** MAML training starts now ***")
    logger.info("*********************************")
    
    loss_fn = torch.nn.CrossEntropyLoss()

    for iteration in tqdm(range(args.num_meta_iterations)): #outer loop
        opt.zero_grad()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_domain_accuracy = 0.0

        tmp_cntr = meta_batch_size_list[:]

        for idx in range(meta_batch_size):
            task = None
            for k in range(10):
                choice_idx = random.randint(0, len(meta_tasks_list)-1)
                if tmp_cntr[choice_idx] > 0:
                    task = next(iter(meta_tasks_list[choice_idx]))
                    tmp_cntr[choice_idx] -= 1
                    break

            n_meta_lr = args.shots // 2

            train_query_inp = {'input_ids': task['input_ids'][:n_meta_lr],
                               'attention_mask': task['attention_mask'][:n_meta_lr],
                               'labels': task['label'][:n_meta_lr]}
            train_support_inp = {'input_ids': task['input_ids'][n_meta_lr:],
                               'attention_mask': task['attention_mask'][n_meta_lr:],
                               'labels': task['label'][n_meta_lr:]}

            train_query_inp = move_to(train_query_inp, device)
            train_support_inp = move_to(train_support_inp, device)

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
            meta_valid_error += eval_loss.item()
            meta_valid_accuracy += eval_acc
            
            if args.exp_setting in ["hmaml-zeroshot", "hmaml-fewshot"]:
                choice_idx = random.randint(0, len(meta_domain_tasks)-1)
                for indx, d_batch in enumerate(meta_domain_tasks):
                    if indx ==  choice_idx:   
                        d_task = d_batch
                domain_query_inp = {'input_ids': d_task['input_ids'],
                    'attention_mask': d_task['attention_mask'],
                    'labels': d_task['label']}
                domain_query_inp = move_to(domain_query_inp, device)
                outputs = learner(domain_query_inp)
                d_loss, d_f1 = compute_loss_acc(outputs["logits"], domain_query_inp["labels"], loss_fn)
                if args.n_gpu > 1:
                    d_loss = d_loss.mean()

                total_loss = eval_loss + d_loss
                meta_domain_accuracy += d_f1

            else:
                total_loss = eval_loss

            total_loss.backward()

        # Print some metrics
        print('\n')
        mt_error = meta_train_error / meta_batch_size
        mt_acc = meta_train_accuracy / meta_batch_size
        mv_error = meta_valid_error / meta_batch_size
        mv_acc =  meta_valid_accuracy / meta_batch_size
        md_acc = meta_domain_accuracy / meta_batch_size
        # mv_error.backward()

        logger.debug('  Iteration {} >> Meta Train Error {:.3f}, F1 {:.3f} # Valid Error {:.3f}, F1 {:.3f}, Domain F1 {:.3f}'.format(iteration, mt_error, mt_acc, mv_error, mv_acc, md_acc))

        #Average the accumulated gradients and optimize
        # for p in maml.parameters():
        #     if p.grad is not None:
        #         p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        if args.exp_setting == "hmaml-zero-refine":
            silver_dataset = get_silver_dataset_for_meta_refine(args, model, target_lang_dataloaders['train'], device)
            meta_tasks_list = [get_dataloader(
                split_name="train", config=args, train_few_dataset_name= f"founta{args.source_lang}", lang=args.source_lang
            ),
            torch.utils.data.DataLoader(
                silver_dataset, batch_size=args.shots, num_workers=args.num_workers, shuffle=True, drop_last = True,
            )]

            meta_batch_size_list = []
            for l in meta_tasks_list:
                meta_batch_size_list.append(len(l))
            meta_batch_size = sum(meta_batch_size_list)
            logger.info(f"Number of meta tasks {meta_batch_size}")


    # Zero-shot, Few-shot or Full-tuned evaluation? You can now take the zero-shot meta-trained model and fine-tune it on 
    # available low-resource few-shot training samples. If we don't fine-tune further, it can be considered as zero-shot model.
    #  
    summary = {"aux_lang": args.aux_lang, 
        "target_lang": args.target_lang, 
        "num_meta_iterations": args.num_meta_iterations,
        "base_model_path": args.base_model_path,
        "script_name": os.path.basename(__file__),
        "exp_setting": args.exp_setting
    }

    ltxt = "Zero-shot" if "zero" in str(args.exp_setting) else "Few-shot"
    logger.info(f"**** {ltxt} evaluation on {args.dataset_name} meta {args.exp_setting} tuned model >>> Language = {args.target_lang} ****")
    zfew_result = evaluate(args, model, target_lang_dataloaders['test'], device, type="pred")
    summary[str(args.exp_setting)] = {"f1": zfew_result[0], "loss": zfew_result[1]}
    
    # **** THE FEW-SHOT FINE-TUNING STEP ****

    if args.exp_setting == "hmaml-zeroshot":
        if args.finetune_fewshot == "few":
            ft_result = finetune(args, model, train_target_ft_dataloader, eval_target_ftdataloader, target_lang_dataloaders['test'], device)
            summary[str(args.finetune_fewshot)] = ft_result
        elif args.finetune_fewshot == "full":
            ft_result = finetune(args, model, target_lang_dataloaders['train'], target_lang_dataloaders['val'], target_lang_dataloaders['test'], device)
            summary[str(args.finetune_fewshot)] = ft_result
    
    os.makedirs(summary_output_dir, exist_ok=True)
    json.dump(summary, open(summary_fname, 'w'))


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

            cols = ['input_ids', 'attention_mask', 'labels']
            silver_batch = dict()
            for col in cols:
                silver_batch[col] = batch[col][bindices].cpu().numpy()
            
            if silver_dataset is None:
                silver_dataset =  silver_batch
            else:
                for key, value in silver_batch.items():
                    silver_dataset[key] = np.concatenate((silver_dataset[key], value), axis=0)

    unique, counts = np.unique(silver_dataset["labels"], return_counts=True)
    lower_bnd = min(min(counts), 200)

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


class SilverDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
    def __getitem__(self, idx):
        item = {k:torch.tensor(v) for k, v in self.data_dict[idx].items()}
        item["label"] = item.pop("labels")
        return item

    def __len__(self):
        return len(self.data_dict)

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
            dataset = HFDataset(
                data_df, config.tokenizer, max_seq_len=config.max_seq_len
            )
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        dataset = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True,
        )

        dataloaders[split_name] = dataset
    return dataloaders


def get_dataloader(split_name, config, train_few_dataset_name=None, lang=None, train="meta"):
    few_pkl_path = os.path.join(
        DEST_DATA_PKL_DIR, f"{train_few_dataset_name}_few_{split_name}.pkl"
    )
    data_df = pd.read_pickle(few_pkl_path, compression=None)
    logger.debug(f"picking {data_df.shape[0]} rows from `{few_pkl_path}` as few samples")

    if lang is not None:
        logger.debug(f"filtering only '{lang}' samples from {split_name} pickle")
        data_df = data_df.query(f"lang == '{lang}'")

    if config.dataset_type == "bert":
        dataset = HFDataset(
            data_df, config.tokenizer, max_seq_len=config.max_seq_len
        )
    else:
        raise ValueError(f"Unknown dataset_type {config.dataset_type}")
    
    if train == "meta":
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.shots, num_workers=config.num_workers, drop_last=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers,
        )

    return dataloader



if __name__ == '__main__':
    args = parse_helper()
    main(args, cuda=True)

"""
A prototype of two step meta learning algorithm for multilingual hate detection.
"""
import argparse
import random
import os
import numpy as np
import pandas as pd
from src.data.consts import SRC_DATA_PKL_DIR
import torch
import learn2learn as l2l

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, \
    AutoTokenizer

from src.data.datasets import HFDataset
from src.data.load import get_dataloader

logging.basicConfig(
                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def parse_helper():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../../data/processed/", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--dataset_type", default="bert", type=str,
                        help="The input data type. It could take bert, lstm, gpt2 as input.")
    parser.add_argument("--dataset_name", default="hateval2019", type=str,
                        help="The name of the task to train selected in the list: [semeval, hateval, hasoc]",
                        )
    # other parameters
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Size of the mini batch")
    parser.add_argument("--base_lang", type=str, default=None, required=True,
                        help="Languages to use for fine-tuning training.")
    parser.add_argument("--meta_train_lang", type=str, default=None, required=True,
                        help="Additional languages to use for meta-training.")
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

def accuracy(predictions, labels):
    preds = np.argmax(predictions, axis=1)
    return (preds == labels).mean()

def report_memory(name=''):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached()/ mega_bytes)
    print(string, end='\r')

def evaluate(args, model, validation_dataloader, device, type="test"):
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch["labels"] = batch.pop("label")
        with torch.no_grad():
            batch = move_to(batch ,device)
            outputs = model(**batch)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            logits = outputs[1]

            logits = logits.detach().cpu().numpy()
            label_ids = batch["labels"].to('cpu').numpy()

            total_eval_loss += loss.item()
            total_eval_accuracy += accuracy(logits, label_ids)

        nb_eval_steps += 1

    val_accuracy = total_eval_accuracy / len(validation_dataloader)
    val_loss = total_eval_loss / len(validation_dataloader)
    log_str = type.capitalize() + "iction" if type == "pred" else type.capitalize() + "uation"

    logger.info("*** Running Model {} **".format(log_str))
    logger.info(f"  Num examples = {len(validation_dataloader)*args.batch_size}")
    logger.info(f"  Accuracy = {val_accuracy:.3f}")
    logger.info(f"  Loss = {val_loss:.3f}")


def train_from_scratch(args, model, opt, scheduler, train_dataloader, eval_dataloader, test_dataloader, device):
    """
    Returns the fine-tuned model which has similar behavior of the pre-trained model e.g. BERT, XLM-R.

    Use this method to fine-tune using high-resouce training samples on a hate detection task. 

    Args:
        model (BERTModelClass):
            The model for task-specific fine-tuning from scratch.

        train_dataloader (:obj:`torch.utils.data.DataLoader`, `optional`):
            The train dataloader to use.
        
        eval_dataloader (:obj:`torch.utils.data.DataLoader`, `optional`):
            The test dataloader to use.

        test_dataloader (:obj:`torch.utils.data.DataLoader`, `optional`):
            The test dataloader to use.

        opt (torch.optim): An optimizer object to use for gradient update.

        scheduler (`get_linear_schedule_with_warmup`): 
            An scheduler to tune the learning rate for BERT training.
            Used in HF transformer codebase.
    
    Returns:
            :class:`~transformers.BertModel`: Returns fine-tuned model e.g. BERT.
    """
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)*args.batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs * len(train_dataloader)}")


    nb_train_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in train_dataloader:
            model.zero_grad()
            batch["labels"] = batch.pop("label")
            batch  = move_to(batch, device)
            outputs = model(**batch)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            logits = outputs[1]

            logits = logits.detach().cpu().numpy()
            label_ids = batch["labels"].to('cpu').numpy()
            acc = accuracy(logits, label_ids)
            loss.backward()
            opt.step()
            scheduler.step()
            total_train_loss += loss.item()
            total_train_acc += acc
            if (nb_train_steps + 1) % 50 == 0:
                report_memory(name = "fine-tune-base")
                avg_train_loss = total_train_loss / nb_train_steps
                logger.info(f"  Epoch {epoch+1}, step {nb_train_steps+1}, training loss: {avg_train_loss:.3f}")
                evaluate(args, model, eval_dataloader, device, type="eval")
                break

            nb_train_steps += 1

    logger.info(f"Evaluating base model performance on test set. Training lang = {args.base_lang}")
    evaluate(args, model, test_dataloader, device, type="pred")
    return model


def main(args,
         meta_lr=2e-3,
         fast_lr=5e-3,
         meta_batch_size=None,
         adaptation_steps=1,
         num_iterations=10,
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
        args.n_gpu = torch.cuda.device_count()
    logger.info(f"device : {device}")
    logger.info(f"gpu : {args.n_gpu}")

    # Download configuration from huggingface.co and cache.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)

    if torch.cuda.device_count() > 0:
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = optim.AdamW(optimizer_grouped_parameters, lr=meta_lr, eps=args.adam_epsilon)

    # Load English samples
    base_lang_dataloaders = get_split_dataloaders(args, lang=args.base_lang)
    train_dataloader, eval_dataloader, test_dataloader = base_lang_dataloaders['train'], base_lang_dataloaders['val'], \
                                                         base_lang_dataloaders['test']

    meta_lang_dataloaders = get_split_dataloaders(args, lang=args.meta_train_lang)
    meta_tasks = meta_lang_dataloaders['val']
    meta_batch_size = len(meta_tasks)

    total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=1, num_training_steps=total_training_steps
    )

    model = train_from_scratch(args, model, opt, scheduler, train_dataloader, eval_dataloader, test_dataloader, device)
    logger.info(f"**** Zero-shot evaluation on before MAML # Lang = {args.meta_train_lang} ****")
    evaluate(args, model, meta_lang_dataloaders['val'], device, type="pred")
    # Step 1 starts from here. This step assumes we have a pretrained base model from `train_from_scratch`, 
    # possibly trained on English. We will now meta-train the base model with MAML algorithm. `Meta-training`
    # support set only contains input from `English` training set. `Meta-training` query set can both contain
    # samples from English and other languages (validation set).

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True)
    logger.info("*** MAML training starts now ***")
    for iteration in tqdm(range(num_iterations)): #outer loop
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for idx, task in enumerate(meta_tasks):
            batch = move_to(task, device)

            n_meta_lr = args.batch_size // 2

            train_query_inp = {'input_ids': batch['input_ids'][:n_meta_lr],
                               'attention_mask': batch['attention_mask'][:n_meta_lr],
                               'labels': batch['label'][:n_meta_lr]}
            train_support_inp = {'input_ids': batch['input_ids'][n_meta_lr:],
                               'attention_mask': batch['attention_mask'][n_meta_lr:],
                               'labels': batch['label'][n_meta_lr:]}

            # train support inp should also contain other languages that we want to meta-adapt in `step two` 

            # Compute meta-training loss
            learner = maml.clone()

            report_memory(name = f"maml{idx}")

            for _ in range(adaptation_steps):
                outputs = learner(**train_support_inp)
                loss = outputs[0]
                logits = outputs[0]
                logger.debug(train_query_inp)
                logger.debug(train_query_inp['labels'].shape)
                logger.debug(logits.shape)

                # logits = logits.detach().cpu().numpy()
                # label_ids = train_support_inp["labels"].to('cpu').numpy()

                # acc = accuracy(logits, label_ids)
                if args.n_gpu > 1:
                    loss = loss.mean()
                learner.adapt(loss, allow_nograd=True, allow_unused=True)
                meta_train_error += loss.item()
                meta_train_accuracy = 10000 

            outputs = learner(**train_query_inp)
            eval_loss = outputs[0]
            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            meta_valid_error += eval_loss.item()
            meta_valid_accuracy = 10000  # need to change later
            eval_loss.backward()
        
        meta_train_error = meta_train_error / meta_batch_size * adaptation_steps
        meta_valid_error = meta_valid_error / meta_batch_size

        # meta_valid_error.backward()
        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()
        opt.zero_grad()

    # Step 2. This section will be updated with Meta-adaption codes. It will only train on 
    # low-resource langauges that we want to adapt to e.g. Spanish. It will follow similar code as
    # above but a little modification on `query` and `support` set preparation (no English samples!). 

    # **** THE LAST STEP ****

    # Zero-shot, Few-shot or Full-tuned evaluation? You can now take the meta-adapted model and fine-tune it on 
    # all the available low-resource training samples. If we don't fine-tune that it can be considered as few-shot model. 
    # If we don't apply step 2, it becomes a zero-shot model.

    logger.info(f"**** Zero-shot evaluation on meta adapted model # Lang = {args.meta_train_lang} ****")
    evaluate(args, model, meta_lang_dataloaders['test'], device, type="pred")


def get_split_dataloaders(config, lang):
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    dataset_name = config.dataset_name
    split_names = ["train", "val", "test"]
    dataloaders = dict()

    for split_name in split_names:
        pkl_path = os.path.join(SRC_DATA_PKL_DIR, f"{dataset_name}{lang}_{split_name}.pkl")
        data_df = pd.read_pickle(pkl_path, compression=None)
        if lang is not None:
            logger.info(f"filtering only '{lang}' samples from {split_name} pickle")
            data_df = data_df.query(f"lang == '{lang}'")
        if config.dataset_type == "bert":
            dataset = HFDataset(
                data_df, config.tokenizer, max_seq_len=config.max_seq_len
            )
        else:
            raise ValueError(f"Unknown dataset_type {config.dataset_type}")

        dataset = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )

        dataloaders[split_name] = dataset
    return dataloaders


if __name__ == '__main__':
    args = parse_helper()
    main(args, cuda=True)

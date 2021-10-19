"""
A prototype of two step meta learning algorithm for multilingual hate detection.
"""
import argparse
import random
import os
import numpy as np
import pandas as pd
import torch
import learn2learn as l2l

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, \
    AutoTokenizer

from src.data.consts import SRC_DATA_PKL_DIR
from src.data.datasets import HFDataset
from src.data.load import get_dataloader

logger = logging.getLogger(__name__)


def parse_helper():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../../data/processed/", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--dataset_type", default="bert", type=str,
                        help="The input data type. It could take bert, lstm, gpt2 as input.")
    parser.add_argument("--task_name", default='semeval', type=str,
                        help="The name of the task to train selected in the list: [semeval, hateval, hasoc]")
    # other parameters
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Size of the mini batch")
    parser.add_argument("--dataset_name", default="hateval2019", type=str,
                        help="Select a dataset for model training",
                        )
    parser.add_argument("--base_lang", type=str, default=None,
                        help="Languages to use for fine-tuning training.")
    parser.add_argument("--meta_train_lang", type=str, default=None,
                        help="Additional languages to use for meta-training.")

    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="The maximum sequence length of the inputs.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for tokenization.")

    args = parser.parse_args()
    logger.info(args)
    return args


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def evaluate(model, validation_dataloader, type="test"):
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch["labels"] = batch.pop("label")

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs[0]
        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = batch["labels"].to('cpu').numpy()

        total_eval_loss += loss.item()
        total_eval_accuracy += accuracy(logits, label_ids)

        nb_eval_steps += 1

    val_accuracy = total_eval_accuracy / len(validation_dataloader)
    val_loss = total_eval_loss / len(validation_dataloader)
    log_str = type.upper() + "iction" if type == "Pred" else "uation"

    print("*** Running Model {} **".format(log_str))
    print("Accuracy: {0:.2f}".format(val_accuracy))
    print("Loss: {0:.2f}".format(val_loss))


def train_from_scratch(args, model, opt, scheduler, epochs, train_dataloader, eval_dataloader, test_dataloader):
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

        epochs (int): The number of training epochs to run.

        opt (torch.optim): An optimizer object to use for gradient update.

        scheduler (`get_linear_schedule_with_warmup`): 
            An scheduler to tune the learning rate for BERT training.
            Used in HF transformer codebase.
    
    Returns:
            :class:`~transformers.BertModel`: Returns fine-tuned model e.g. BERT.
    """

    nb_train_steps = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in train_dataloader:
            model.zero_grad()
            batch["labels"] = batch.pop("label")
            outputs = model(**batch)
            loss = outputs[0]
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = batch["labels"].to('cpu').numpy()
            acc = accuracy(logits, label_ids)
            loss.backward()
            opt.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_acc += acc
            if (nb_train_steps + 1) % 100 == 0:
                avg_train_loss = total_train_loss / nb_train_steps
                print("epoch {}, step {}, training loss: {0:.2f}".format(epoch + 1, nb_train_steps, avg_train_loss))
                evaluate(model, eval_dataloader, type="eval")

            nb_train_steps += 1

    print(f"Evaluating base model performance on test set. Training lang = {args.base_lang}")
    evaluate(model, test_dataloader, type="pred")
    return model


def main(args,
         meta_lr=0.003,
         fast_lr=0.5,
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
        device = torch.device('cuda:{}'.format(3))

    num_train_epochs = 3
    train_dataloader = 1000  # dummy
    gradient_accumulation_steps = 1  # dummy

    # Download configuration from huggingface.co and cache.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)
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
    meta_batch_size = meta_lang_dataloaders['val']

    total_training_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=1, num_training_steps=total_training_steps
    )

    train_from_scratch(args, model, opt, scheduler, num_train_epochs, train_dataloader, eval_dataloader, test_dataloader)
    print(f"**** Zero-shot evaluation on before MAML # Lang = {args.meta_train_lang} ****")

    # Step 1 starts from here. This step assumes we have a pretrained base model from `train_from_scratch`, 
    # possibly trained on English. We will now meta-train the base model with MAML algorithm. `Meta-training`
    # support set only contains input from `English` training set. `Meta-training` query set can both contain
    # samples from English and other languages (validation set).

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)

    for iteration in tqdm(num_iterations):
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in meta_batch_size:
            batch = tuple(t.to(device) for t in task)

            n_meta_lr = args.batch_size // 2

            train_query_inp = {'input_ids': batch[0][:n_meta_lr],
                               'attention_mask': batch[1][:n_meta_lr],
                               'labels': batch[3][:n_meta_lr]}
            train_support_inp = {'input_ids': batch[0][n_meta_lr:],
                                 'attention_mask': batch[1][n_meta_lr:],
                                 'labels': batch[3][n_meta_lr:]}

            # train support inp should also contain other languages that we want to meta-adapt
            # in step two 

            # Compute meta-training loss
            learner = maml.clone()
            for _ in range(adaptation_steps):
                outputs = learner(**train_support_inp)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()
                learner.adapt(loss, allow_nograd=True, allow_unused=True)
                meta_train_error += loss
                meta_train_accuracy = 10000  # need to change later

            outputs = learner(**train_query_inp)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            meta_valid_error += loss
            meta_valid_accuracy = 10000  # need to change later

        meta_valid_error = meta_valid_error / meta_batch_size

        meta_valid_error.backward()
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

    # Step 2. This section will be update with Meta adaption codes. It will only train on 
    # low-resource langauges that we want to adapt to e.g. Spanish. It will follow similar code as
    # above but a little modification (no English samples!). 

    # **** THE LAST STEP ****

    # Zero-shot, Few-shot or Full-tuned evaluation? You can now take this model and fine-tune it on full low-resource 
    # training samples. If we don't fine-tune that it can be considered as few-shot model. If we don't apply step 2
    # it becomes a zero-shot model.

    print(f"**** Zero-shot evaluation on meta adapted model # Lang = {args.meta_train_lang} ****")
    evaluate(model, meta_lang_dataloaders['test'], type="pred")


def get_split_dataloaders(config, lang):
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    dataset_name = config.dataset_name
    split_names = ["train", "val", "test"]
    dataloaders = dict()

    for split_name in split_names:
        pkl_path = os.path.join("../../" + SRC_DATA_PKL_DIR, f"{dataset_name}{lang}_{split_name}.pkl")
        data_df = pd.read_pickle(pkl_path, compression=None)
        if lang is not None:
            print(f"filtering only '{lang}' samples from {split_name} pickle")
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

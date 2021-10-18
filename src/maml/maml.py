"""
A prototype of two step meta learning algorithm for multilingual hate detection.
"""
import argparse
import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


def parse_helper():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../../data/processed/", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default='semeval', type=str,
                        help="The name of the task to train selected in the list: [semeval, hateval, hasoc]")
    # other parameters
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    args = parser.parse_args()
    logger.info(args)
    return args


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def main(args,
         meta_lr=0.003,
         fast_lr=0.5,
         meta_batch_size=32,
         adaptation_steps=1,
         num_iterations=60000,
         cuda=False,
         seed=42, ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    num_train_epochs = 3
    train_dataloader = 1000  # dummy
    gradient_accumulation_steps = 1  # dummy

    # Download configuration from huggingface.co and cache.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=3)
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    total_training_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = optim.AdamW(optimizer_grouped_parameters, lr=meta_lr, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=1, num_training_steps=total_training_steps
    )

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in tqdm(num_train_epochs):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in meta_batch_size:
            batch = tuple(t.to(device) for t in task)

            train_query_inp = ""
            train_support_inp = ""
            test_query_inp = ""
            test_support_inp = ""

            # Compute meta-training loss
            learner = maml.clone()
            for _ in range(adaptation_steps):
                outputs = learner(**train_support_inp)
                loss = outputs[0]
                loss = loss.mean()
                learner.adapt(loss, allow_nograd=True, allow_unused=True)
                meta_train_error += loss

            outputs = learner(**train_query_inp)
            loss = outputs[0]
            loss = loss.mean()
            meta_valid_error += loss
            # Print some metrics

        meta_valid_error.backward()
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()


if __name__ == '__main__':
    args = parse_helper()
    # main(args)

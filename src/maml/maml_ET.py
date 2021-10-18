import argparse
import random
import numpy as np
import torch
import learn2learn as l2l
from src.data.load import get_dataloader
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
from src.config import EmptyConfig
logger = logging.getLogger(__name__)


def parse_helper():
    parser = argparse.ArgumentParser()

    # Needed to be Feed

    parser.add_argument(
        "--lang_aug",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="should be one of: xlmr, mbert, lstm",
    )
    parser.add_argument("--batch",
                        default=64,
                        type=int,
                        help="Size of the Batch")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        # help="",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
        # help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        # help="",
    )
    parser.add_argument(
        "--maml_lr",
        type=float,
        default=2e-3,
        # help="",
    )


    # Required parameters
    parser.add_argument("--data_dir",
                        default="../../data/processed/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--path_to_model",
                        default='bert-base-multilingual-cased',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name",
                        default='semeval',
                        type=str,
                        help="The name of the task to train selected in the list: [semeval, hateval, hasoc]")


    # other parameters
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    args = parser.parse_args()
    logger.info(args)
    return args


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def compute_loss(model,iterator,device):


    # Include Acc part Later

    loss=0.0
    acc=0.0

    for batch in tqdm(iterator):

        input_ids=batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        cur_loss=outputs[0].mean()
        loss+=cur_loss

    loss/=len(iterator)

    return  loss




def main(args,
         num_tasks=1,
         num_shots=1,
         ,seed=32):
    config = EmptyConfig()

    config.epochs = args["epochs"]
    config.path_to_model=args['path_to_model']
    config.hp.batch_size = args["batch_size"]
    config.hp.max_seq_len = args["max_seq_len"]

    config.lang=args['lang_aug'] # if we were to run in multiple languages this needs editing

    config.hp.lr=args["lr"]
    config.hp.maml_lr = args["maml_lr"]

    if args["model_type"] in ["xlmr", "mbert"]:
        config.dataset_type = "bert"
        if args["model_type"] == "xlmr":
            config.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        else:
            config.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-uncased"
            )
    else:
        raise ValueError(f"Unknown model_type")


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset=get_dataloader(args.dataset_name,"train",config)
    train_dataset_gen=DataLoader(train_dataset,shuffle=True,batch_size=config.hp.batch_size)

    val_dataset = get_dataloader(args.dataset_name, "val", config)
    val_dataset_gen = DataLoader(train_dataset, shuffle=True, batch_size=config.hp.batch_size)

    # If we more than one language as aux then to feed this too
    """mlm_gen=l2l.data.UnionMetaDataset(
        mlm_dataset,
        num_tasks=num_tasks,
    )"""

    # Download configuration from huggingface.co and cache.
    config = AutoConfig.from_pretrained(config.path_to_model, num_labels=3)
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
    opt = optim.AdamW(optimizer_grouped_parameters, lr=config.hp.lr, eps=args.adam_epsilon)


    mamal_model = l2l.algorithms.MAML(model, lr=config.hp.maml_lr, first_order=False)

    combined_loss=None

    for n_epoch in range(1): #Number of epoch is 1
        for n_task in range(1): # for now assuming only one task

            task_model=mamal_model.clone()

            for n_shot in range(num_shots):
                adaptation_loss=compute_loss(task_model,train_dataset_gen,device)
                task_model.adapt(adaptation_loss)
                print("For Task {} and shot {} ada_loss is".format(num_tasks+1,num_shots+1,adaptation_loss))

            evaluation_loss=compute_loss(task_model,val_dataset_gen,device) #this needs to change if num task in one iteration changes
            print("For Task {} eval_loss is".format(num_tasks + 1, evaluation_loss))

            opt.zero_grad()
            evaluation_loss.backward()
            opt.step()

    # Save Model







if __name__ == '__main__':
    args = parse_helper()
    # main(args)

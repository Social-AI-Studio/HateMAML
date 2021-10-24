import sys

sys.path.append(".")

import argparse
import datetime
import random
import numpy as np
import torch
import learn2learn as l2l
from src.data.load import get_dataloader
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from src.config import EmptyConfig
from src.model.lightning import LitClassifier
import os
logger = logging.getLogger(__name__)
from src.model.classifiers import MBERTClassifier, XLMRClassifier
from src.data.consts import RUN_BASE_DIR

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
    print(string)

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
        "--num_iter",
        type=int,
        default=5,
        help="Number of iterations to run x-mamal",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="should be one of: xlmr, mbert, lstm",
    )
    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Size of the Batch")
    
    parser.add_argument("--max_seq_len",
                        default=64,
                        type=int,
                        help="Length of Sequence")
   

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        # help="",
    )
    parser.add_argument(
        "--maml_lr",
        type=float,
        default=5e-6,
        # help="",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        # help="",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        # help="",
    )

    # Required parameters
    parser.add_argument("--dataset_name",
                        type=str,
                        help="The input data name example hateval2019en")
    parser.add_argument("--path_to_model",
                        default='bert-base-multilingual-cased',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    
   
    args = parser.parse_args()
    logger.info(args)
    return vars(args)


def checkpoint_model(model,epoch,aux_lang,run_name,):
    run_dir = os.path.join(RUN_BASE_DIR,"xmamal",run_name)
    os.makedirs(run_dir)
    save_path = os.path.join(run_dir,f"Iterations={epoch}_aux_lang={aux_lang}.ckpt")
    torch.save(model.state_dict(), save_path) 


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def compute_loss(model,batch,loss,device):


    # Include Acc part Later
    outputs = model(batch)

    pred_labels = torch.argmax(outputs["logits"], dim=1)
    actual_labels = batch['labels']

    loss = loss(outputs["logits"], actual_labels)
    acc = (pred_labels == actual_labels).sum() / pred_labels.shape[0]

    return  loss,acc




def main(args,
         seed=32):
    config = EmptyConfig()

    #config.epochs = args["epochs"]
    config.dataset_name=args['dataset_name']
    config.path_to_model=args['path_to_model']
    config.hp.batch_size = args["batch_size"]
    config.hp.max_seq_len = args["max_seq_len"]
    config.hp.num_iter=args['num_iter']

    config.lang=args['lang_aug'] # if we were to run in multiple languages this needs editing

    config.hp.lr=args["lr"]
    config.hp.maml_lr = args["maml_lr"]

    config.num_workers = args['num_workers']
    config.hp.dropout = args['dropout']
    
    print("_____LOADING MODEL AND TOKENISER_____")

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

    if args["model_type"] == "mbert":
        model = MBERTClassifier(config)
    else:
        raise ValueError(f"Classifier not available")


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset=get_dataloader(config.dataset_name,"train",config)

    val_dataset = get_dataloader(config.dataset_name, "val", config)


    # If we more than one language as aux then to feed this too
#     mlm_gen=l2l.data.UnionMetaDataset(
#        mlm_dataset,
#        num_tasks=num_tasks,
#    )"""

    print("Loading Model")
    lit_model =LitClassifier(model, config)
    ckpt = torch.load(os.path.normpath(config.path_to_model))
    lit_model.load_state_dict(ckpt['state_dict'])
    model = lit_model.model
    model.to(device)
    
    
    mamal_model = l2l.algorithms.MAML(model, lr=config.hp.maml_lr, first_order=True,allow_unused=True)
    
    opt = optim.Adam(mamal_model.parameters(), lr=config.hp.lr)
    print("Model Loaded")
    

    
    model_loss = torch.nn.CrossEntropyLoss()

    list_of_task=[]
    list_of_task.append(val_dataset)
    num_support=int(0.5*config.hp.batch_size)

    run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")



    for n_epoch in range(config.hp.num_iter):
        
        print("{} Epoch/iteration".format(n_epoch))

        iteration_error=0.0
        n_pseudo_task = 0
        for i,aux_task in enumerate(list_of_task): # Create batch over validation set creating a set of fake task
            print("Language {} is being seen".format(i))
            task_model=mamal_model.clone()

            for batch in tqdm(aux_task): # iterate over adaptation set

                input_fast_train = {'input_ids': batch['input_ids'][:num_support].to(device),
                                    'attention_mask': batch['attention_mask'][:num_support].to(device),
                                    'labels': batch['label'][:num_support].to(device)}

                adaptation_loss,_=compute_loss(task_model,input_fast_train,model_loss,device)
                
                task_model.adapt(adaptation_loss, allow_nograd=True, allow_unused=True) 
                print("For psedo_task {} ada_loss is {}".format(n_pseudo_task+1,adaptation_loss))

                input_fast_val = {'input_ids': batch['input_ids'][num_support:].to(device),
                                    'attention_mask': batch['attention_mask'][num_support:].to(device),
                                    'labels': batch['label'][num_support:].to(device)}

                evaluation_loss,_=compute_loss(task_model,input_fast_val,model_loss,device) 
                
                print("For psedo_task {} adapt_eval_loss is {}".format(n_pseudo_task + 1, evaluation_loss))
                n_pseudo_task+=1
                
                
                
                evaluation_loss.backward()
        opt.step()
        mamal_model.zero_grad()
        
    print("Save Model")
    checkpoint_model(mamal_model,config.hp.num_iter,config.lang,run_name+"_mamal_model"+"_"+config.dataset_name)


    # Save Model add code









if __name__ == '__main__':
    args = parse_helper()
    main(args)

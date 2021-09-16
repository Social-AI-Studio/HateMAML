# An example baseline to give an idea how the project is structured.
import sys

sys.path.append(".")

from src.config import EmptyConfig
from src.data.load import get_dataloader
from transformers import AutoTokenizer

if __name__ == "__main__":

    # config can be initialized with default instead of empty values.
    config = EmptyConfig()

    # config can be populated from commandline arguments here. hardcoding
    # for this example.
    config.dataset_type = "bert"
    config.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    config.batch_size = 32
    config.max_seq_len = 32
    config.num_workers = 8

    train_dataloader = get_dataloader(
        dataset_name="hateval2019", split_name="train", config=config
    )
    print(f"successfully initialized dataloader of {len(train_dataloader)} batches")

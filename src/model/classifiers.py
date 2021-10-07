import torch
from transformers import XLMRobertaModel


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super(ClassificationHead, self).__init__()
        self.ff1 = torch.nn.Linear(in_dim, in_dim // 2)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.ff2 = torch.nn.Linear(in_dim // 2, out_dim)

    def forward(self, in_tensor):
        return self.ff2(self.dropout(self.ff1(in_tensor)))


class XLMRClassifier(torch.nn.Module):
    def __init__(self, config):
        super(XLMRClassifier, self).__init__()
        self.lm = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.classification_head = ClassificationHead(
            self.lm.config.hidden_size, 2, config.hp.dropout
        )

    def forward(self, batch):
        lm_out_dict = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        logits = self.classification_head(lm_out_dict["pooler_output"])
        return {"logits": logits}

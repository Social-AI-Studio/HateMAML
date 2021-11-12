import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch


class LitClassifier(pl.LightningModule):
    """
    A wrapper class to facilitate training through pytorch lightning.
    """

    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        self.lr = config.hp.lr if config is not None else 2e-6
        self.criterion = torch.nn.CrossEntropyLoss()

    def reinitialise_head(self):
        self.model.reinitialise_head()

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = trainable

    def set_freeze_layers(self,freezing_mode):
        return self.model.set_freeze_layers(freezing_mode)

    def shared_step(self, batch, batch_idx, return_pred_labels=False):
        """
        this step will be shared by the train/val/test logic
        """
        out_dict = self.model(batch)
        #        out_dict = self.model(
        #            input_ids=batch["input_ids"],
        #            attention_mask=batch["attention_mask"],
        #            return_dict=True,
        #        )

        pred_labels = torch.argmax(out_dict["logits"], dim=1)
        actual_labels = batch["label"]

        loss = self.criterion(out_dict["logits"], actual_labels)

        # Logging to TensorBoard by default
        metrics = {}

        pred_labels = pred_labels.detach().cpu().numpy()
        actual_labels = actual_labels.detach().cpu().numpy()
        logits = out_dict["logits"].detach().cpu().numpy()

        metrics["loss"] = loss
        metrics["acc"] = (pred_labels == actual_labels).sum() / pred_labels.shape[0]
        metrics["macro_f1"] = f1_score(actual_labels, pred_labels, average="macro")
        metrics["weighted_f1"] = f1_score(
            actual_labels, pred_labels, average="weighted"
        )
        return_dict = {"loss": loss, "metrics": metrics}
        if return_pred_labels:
            return_dict["pred_labels"] = pred_labels
        return return_dict

    def training_step(self, batch, batch_idx):
        shared_step_out_dict = self.shared_step(batch, batch_idx)
        loss, metrics = shared_step_out_dict["loss"], shared_step_out_dict["metrics"]

        metrics = {"train_" + k: v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        shared_step_out_dict = self.shared_step(batch, batch_idx)
        loss, metrics = shared_step_out_dict["loss"], shared_step_out_dict["metrics"]

        metrics = {"val_" + k: v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        shared_step_out_dict = self.shared_step(batch, batch_idx)
        loss, metrics = shared_step_out_dict["loss"], shared_step_out_dict["metrics"]

        metrics = {"test_" + k: v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        defines optimizers and LR schedulers to be used by the trainer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

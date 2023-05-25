import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification
import torchmetrics


class BaselineBERT(pl.LightningModule):
    """
    This class was taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    def __init__(
            self,
            learning_rate,
            num_labels,
            model_name,
            weight_decay,
    ):
        super().__init__()
        self.tag = "baseline BERT"
        self.criterion=torch.nn.CrossEntropyLoss()
        
        # WANDB settings - log hyperparameters
        self.save_hyperparameters()


        # introduce BERT model or ity si fats derivate
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(num_classes=num_labels)
        self.valid_acc = torchmetrics.Accuracy(num_classes=num_labels)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_labels)
        print("Model " + self.tag + " initiated")

    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            token_type_ids = batch.get("token_type_ids"),
            labels=batch["label"]
        )
        preds = outputs.logits.argmax(dim=1).view(-1, 1)

        loss = outputs.loss


        self.log('train/loss', loss, on_epoch=True)
        self.log('train/accuracy', self.train_acc(preds, batch["label"]), on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            token_type_ids = batch.get("token_type_ids"),
            labels=batch["label"]
        )
        preds = outputs.logits.argmax(dim=1).view(-1, 1)

        loss = outputs.loss

        self.log("validation/loss", loss, on_epoch=True) #on_step=False
        self.log("validation/accuracy", self.valid_acc(preds, batch["label"]), on_epoch=True) #on_step=False

        return preds


    def test_step(self, batch, batch_idx):

        outputs = self.forward(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            token_type_ids = batch.get("token_type_ids"),
            labels=batch["label"]
        )
        
        preds = outputs.logits.argmax(dim=1).view(-1, 1)

        loss = outputs.loss

        self.log("test/loss", loss, on_epoch=True) #on_step=False
        self.log("test/accuracy", self.test_acc(preds, batch["label"]), on_epoch=True) #on_step=False

        return preds
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)
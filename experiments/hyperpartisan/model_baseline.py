import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification
import torchmetrics
from sklearn import metrics


class BaselineBERT(pl.LightningModule):
    """
    This class was taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    def __init__(
            self,
            learning_rate,
            num_labels,
            model_name,
            weight_decay=0.01,
            f1_log = False,
            metrics_type = "macro",

    ):
        super().__init__()
        self.tag = "baseline BERT"
        self.criterion=torch.nn.CrossEntropyLoss()
        
        # WANDB settings - log hyperparameters
        self.save_hyperparameters()

        self.train_class_weights = None
        self.model_name = model_name
        # introduce BERT model or ity si fats derivate
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(num_classes=num_labels)
        self.valid_acc = torchmetrics.Accuracy(num_classes=num_labels)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_labels)

        # F1 score related variables
        self.f1_aggr = metrics_type
        self.f1 = f1_log
        self.validation_labels = []
        self.validation_predictions = []
        self.test_labels = []
        self.test_predictions = []

        print("Model " + self.tag + " initiated")

    
    def forward(self, **inputs):
        if self.model_name == "allenai/longformer-base-4096":
            global_attention_mask = inputs["attention_mask"].clone()
            global_attention_mask[:, 0] = 1  # Set global attention on the [CLS] token

            # Update the inputs dictionary with the global attention mask
            inputs["global_attention_mask"] = global_attention_mask
        return self.model(**inputs)
    
    def set_train_class_weights(self, weights):
        self.train_class_weights = weights

    def _common_step(self, batch, batch_idx, phase):

        outputs = self.forward(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            token_type_ids = batch.get("token_type_ids"),
            labels=batch["label"]
        )
        preds = outputs.logits.argmax(dim=1).view(-1, 1)
        accuracy = self.train_acc(preds, batch["label"])

        if self.train_class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                batch["label"].squeeze(dim=1),
                reduction="mean", 
                weight=self.train_class_weights
            )
        else:
            loss = outputs.loss

        self.log(f"{phase}/loss", loss, on_epoch=True)
        self.log(f"{phase}/accuracy", accuracy, on_epoch=True)

        return loss, self.train_acc(preds, batch["label"]), preds

    def training_step(self, batch, batch_idx):
        loss, acc, preds = self._common_step(batch, batch_idx, "train")

        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, acc, preds = self._common_step(batch, batch_idx, "validation")

        if self.f1:
            self.validation_labels.append(batch["label"])
            self.validation_predictions.append(preds)

        return preds


    def test_step(self, batch, batch_idx):
        loss, acc, preds = self._common_step(batch, batch_idx, "test")
        
        if self.f1:
            self.test_labels.append(batch["label"])
            self.test_predictions.append(preds)

        return preds
    
    def validation_epoch_end(self, validation_step_outputs):
        
        if self.f1:
            labels = torch.vstack(self.validation_labels).cpu()
            predictions = torch.vstack(self.validation_predictions).cpu()

            f1_score = metrics.f1_score(labels, predictions, average=self.f1_aggr)
            recall_score = metrics.recall_score(labels, predictions, average=self.f1_aggr)
            precision_score = metrics.precision_score(labels, predictions, average=self.f1_aggr)

            self.log("validation/f1_score", f1_score)
            self.log("validation/recall_score", recall_score)
            self.log("validation/precision_score", precision_score)

        self.validation_labels = []
        self.validation_predictions = []

    
    def test_epoch_end(self, test_step_outputs):
        
        if self.f1:
            labels = torch.vstack(self.test_labels).cpu()
            predictions = torch.vstack(self.test_predictions).cpu()

            f1_score = metrics.f1_score(labels, predictions, average=self.f1_aggr)
            recall_score = metrics.recall_score(labels, predictions, average=self.f1_aggr)
            precision_score = metrics.precision_score(labels, predictions, average=self.f1_aggr)

            self.log("test/f1_score", f1_score)
            self.log("test/recall_score", recall_score)
            self.log("test/precision_score", precision_score)

        self.test_labels = []
        self.test_predictions = []

    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)
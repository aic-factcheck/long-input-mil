import pytorch_lightning as pl
import torch
from transformers import AutoModel
import torchmetrics
import numpy as np
from sklearn import metrics

class BERTPlusMILModel(pl.LightningModule):
    """
    The evaluation and step functions in this class are inspired and taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    def __init__(
        self,
        prepNN,
        afterNN,
        aggregation_func,
        num_labels = 2,
        model_name = "bert-base-cased",
        learning_rate = 0.001,
        weight_decay = 0.01,
        freeze_params = False,
    ):
        super().__init__()
        self.tag = "E2E MIL"
        self.aggregation_func = aggregation_func
        self.prepNN = prepNN
        self.afterNN = afterNN
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        #self.criterion = torch.nn.CrossEntropyLoss()

        #WANDB settings - log hyperparameters
        self.save_hyperparameters()


        # introduce BERT model or ity si fats derivate
        self.model = AutoModel.from_pretrained(model_name)

        # freeze parameters in BERT model
        if freeze_params:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        self.train_loss = []
        self.train_accuracy = []
        self.last_log_step = self.global_step

        self.validation_labels = []
        self.validation_predictions = []
        self.test_labels = []
        self.test_predictions = []


        # compute the accuracy 
        self.train_acc = torchmetrics.Accuracy(num_classes=2)
        self.valid_acc = torchmetrics.Accuracy(num_classes=2)
        self.test_acc = torchmetrics.Accuracy(num_classes=2)
        self.f1 = True

        print("Model " + self.tag + " initiated")

    def label_decide(self,output):
        return torch.argmax(output, dim=1) #changed it from dim=1 to dim=0 
    
    def set_train_class_weights(self, weights):
        self.train_class_weights = weights

    def forward(self, ids_bert, attention_mask, token_type_ids, batch_ids):
        if token_type_ids is not None: # some Transformer models do not use token_type ids
            bert_output = self.model(
                input_ids = ids_bert,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids, 
                output_hidden_states=True, #CLS tokens
            )
        else:
            bert_output = self.model(
                input_ids = ids_bert,
                attention_mask = attention_mask,
                output_hidden_states=True, #CLS tokens
            )

        #export embedded CLS token
        embeddings = bert_output.last_hidden_state[:,0,:] 

        ids =  batch_ids #bag ids of each instance
        x = embeddings        
        device = x.device

        #preprocess embeddinngs  
        NN_out = self.prepNN(x)
        
        unique, inverse, counts = torch.unique(ids, sorted = True, return_inverse = True, return_counts = True)
        output = torch.empty((len(unique), len(NN_out[0])), device = device)
        
        # aggregate over bag, ids are bag ids
        for i in range(len(unique)):
            output[i] = self.aggregation_func(NN_out[ids == i], dim = 0)
        
        # postprocessing layer
        output = self.afterNN(output)

        return output
    

    
    def _compute_loss_accuracy(self, probs, predictions, targets, weight=None):
        assert (
            predictions.shape == targets.shape
        ), f"predictions and targets have different shapes, {predictions.shape} and {targets.shape}"

        loss = torch.nn.functional.cross_entropy(probs, targets, reduction="mean", weight=weight)
        accuracy = (predictions == targets).float().mean().item()

        return loss, accuracy
    
    @staticmethod
    def _compute_micro_macro(y_true, y_pred):
        scores = dict()
        for avg in ["micro", "macro"]:
            scores[avg]["f1"] = metrics.f1_score(y_true, y_pred, average=avg)
            scores[avg]["recall"] = metrics.recall_score(y_true, y_pred, average=avg)
            scores[avg]["precision"] = metrics.precision_score(y_true, y_pred, average=avg)

        return scores

    
    def _common_step(self, batch, batch_idx, phase):
        batch_idx = batch['batch_idx']
        input_idx = batch["input_ids"]
        if "token_type_ids" in batch:
            token_type = batch["token_type_ids"]
        else:
            token_type = None
        att_mask = batch["attention_mask"]
        lbls = batch['label']
        lbls = lbls.type(torch.LongTensor).to(self.device)
        
        outputs = self.forward(batch_ids=batch_idx, ids_bert=input_idx, token_type_ids=token_type, attention_mask=att_mask)

        pred = self.label_decide(outputs)

        loss, accuracy = self._compute_loss_accuracy(
            outputs, pred, batch["label"].squeeze(dim=1), #maybe delete torch.squeeze()
            weight=self.train_class_weights if phase in ["train", "validation"] else None
        )

        if phase is not None and phase != "train":
            self.log(f"{phase}/loss", loss, batch_size=batch["label"].shape[0])
            self.log(f"{phase}/accuracy", accuracy, batch_size=batch["label"].shape[0])

        return loss, accuracy, pred

    def training_step(self, batch, batch_idx):
        loss, accuracy, pred = self._common_step(batch, batch_idx, phase="train")

        # Specific logging of the training metrics
        # Average the metrics since last logged optimization step
        if self.last_log_step != self.global_step:
            self.log("train/loss", np.mean(self.train_loss), on_epoch=True)
            self.log("train/accuracy", np.mean(self.train_accuracy), on_epoch=True)
            self.train_loss = []
            self.train_accuracy = []
            self.last_log_step = self.global_step
        else:
            self.train_loss.append(loss.item())
            self.train_accuracy.append(accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, predictions = self._common_step(batch, batch_idx, phase="validation")

        self.validation_labels.append(batch["label"])
        self.validation_predictions.append(predictions)

        return predictions
    
    def validation_epoch_end(self, validation_step_outputs):
        if self.f1:
            labels = torch.vstack(self.validation_labels).cpu()
            predictions = torch.vstack(self.validation_predictions).cpu()
            labels = labels.squeeze()
            predictions = predictions.view(-1)
            
            self.log("validation/f1_score.micro", metrics.f1_score(labels, predictions, average="micro"))
            self.log("validation/recall_score.micro", metrics.recall_score(labels, predictions, average="micro"))
            self.log("validation/precision_score.micro", metrics.precision_score(labels, predictions, average="micro"))

            self.log("validation/f1_score.macro", metrics.f1_score(labels, predictions, average="macro"))
            self.log("validation/recall_score.macro", metrics.recall_score(labels, predictions, average="macro"))
            self.log("validation/precision_score.macro", metrics.precision_score(labels, predictions, average="macro"))

        self.validation_labels = []
        self.validation_predictions = []


    def test_step(self, batch, batch_idx):
        loss, accuracy, predictions = self._common_step(batch, batch_idx, phase="test")

        self.test_labels.append(batch["label"])
        self.test_predictions.append(predictions)

        return predictions
    
    def test_epoch_end(self, test_step_outputs):
        if self.f1:
            self.test_labels = self.test_labels[:-1]
            self.test_predictions = self.test_predictions[:-1]
            print("test labels", self.test_labels)
            print("test labels", self.test_predictions)
            labels = torch.vstack(self.test_labels).cpu()
            predictions = torch.vstack(self.test_predictions).cpu()
            labels = labels.squeeze()
            predictions = predictions.view(-1)
            
            self.log("test/f1_score.micro", metrics.f1_score(labels, predictions, average="micro"))
            self.log("test/recall_score.micro", metrics.recall_score(labels, predictions, average="micro"))
            self.log("test/precision_score.micro", metrics.precision_score(labels, predictions, average="micro"))

            self.log("test/f1_score.macro", metrics.f1_score(labels, predictions, average="macro"))
            self.log("test/recall_score.macro", metrics.recall_score(labels, predictions, average="macro"))
            self.log("test/precision_score.macro", metrics.precision_score(labels, predictions, average="macro"))

        self.test_labels = []
        self.test_predictions = []


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

def aggregation_max(x, dim):
  return torch.max(x, dim)[0]

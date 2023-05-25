"""
DISCLAIMER
This is a general concept of this class used for experimenting with MIL classifier proposed in 
thesis Multiple Instance Learning for Long Input NLP Models by Vojtech Jerabek.

File is not executable as it is, additional features whcih are experiments specific need to be implemented.
Example of implementations are in experiments folder where are source files for all executed experiments published in the thesis.
"""

import pytorch_lightning as pl
import torch
from transformers import AutoModel
import torchmetrics

class BERTPlusMILModel(pl.LightningModule):
    def __init__(
        self,
        prepNN, #preprocessing layer
        afterNN, #postprocessing layer
        aggregation_func,
        num_classes = 2,
        model_name = "bert-base-cased",
        learning_rate = 0.00002,
        weight_decay = 0.01,
        freeze_params = False,
        input_size = 768,
    ):
        super().__init__()
        self.tag = "E2E MIL"
        self.aggregation_func = aggregation_func
        self.prepNN = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1268), 
            torch.nn.ReLU(),
            torch.nn.Linear(1268,634),
            torch.nn.ReLU(),
            torch.nn.Linear(634,256),
            torch.nn.ReLU(),)

        self.afterNN = torch.nn.Sequential( 
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes), )

        self.criterion = torch.nn.CrossEntropyLoss()

        #WANDB settings - log hyperparameters
        self.save_hyperparameters()

        # import language model
        self.model = AutoModel.from_pretrained(model_name)

        # freeze parameters in model
        if freeze_params:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        # compute the accuracy
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)

        print("Model " + self.tag + " initiated")

    def label_decide(self,output):
        return torch.argmax(output, dim=1).float() 

    def forward(self, ids_bert, attention_mask, token_type_ids, batch_ids):
        
        bert_output = self.model(
            input_ids = ids_bert,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids, 
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
        
        #postprocessing layer
        output = self.afterNN(output)

        return output

    def training_step(self, batch, batch_idx):
        batch_idx = batch['batch_idx']
        input_idx = batch["input_ids"]
        token_type = batch["token_type_ids"]
        att_mask = batch["attention_mask"]
        lbls = batch['label']
        lbls = lbls.type(torch.LongTensor).to(self.device)
        
        pred = self(batch_ids=batch_idx, ids_bert=input_idx, token_type_ids=token_type, attention_mask=att_mask)
        
        loss = self.criterion(pred, lbls.view(-1))
        pred = self.label_decide(pred)
        
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/accuracy', self.train_acc(pred, lbls.view(-1).int()), on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_idx = batch['batch_idx']
        input_idx = batch["input_ids"]
        token_type = batch["token_type_ids"]
        att_mask = batch["attention_mask"]
        lbls = batch['label']
        lbls = lbls.type(torch.LongTensor).to(self.device)

        pred = self(batch_ids=batch_idx, ids_bert=input_idx, token_type_ids=token_type, attention_mask=att_mask)
        
        loss = self.criterion(pred, lbls.view(-1))
        pred = self.label_decide(pred)
        
        self.log("validation/loss", loss, on_epoch=True) 
        self.log("validation/accuracy", self.valid_acc(pred, lbls.view(-1).int()), on_epoch=True) 

        return loss

    def test_step(self, batch, batch_idx):
        batch_idx = batch['batch_idx']
        input_idx = batch["input_ids"]
        token_type = batch["token_type_ids"]
        att_mask = batch["attention_mask"]
        lbls = batch['label']
        lbls = lbls.type(torch.LongTensor).to(self.device)

        pred = self(batch_ids=batch_idx, ids_bert=input_idx, token_type_ids=token_type, attention_mask=att_mask)
        
        loss = self.criterion(pred, lbls.view(-1)) 
        pred = self.label_decide(pred)
        
        self.log("test/loss", loss, on_epoch=True) 
        self.log("test/accuracy", self.test_acc(pred, lbls.view(-1).int()), on_epoch=True) 

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

def aggregation_max(x, dim):
  return torch.max(x, dim)[0]

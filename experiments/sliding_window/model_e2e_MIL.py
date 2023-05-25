import pytorch_lightning as pl
import torch
from transformers import AutoModel
import torchmetrics

class BERTPlusMILModel(pl.LightningModule):
    def __init__(
        self,
        prepNN,
        afterNN,
        aggregation_func,
        num_classes = 2,
        model_name = "bert-base-cased",
        learning_rate = 0.00002,
        weight_decay = 0.01,
        freeze_params = False,
    ):
        super().__init__()
        self.tag = "E2E MIL"
        self.aggregation_func = aggregation_func
        self.prepNN = prepNN
        self.afterNN = afterNN
        self.criterion = torch.nn.CrossEntropyLoss()

        #WANDB settings - log hyperparameters
        self.save_hyperparameters()


        # introduce BERT model or ity si fats derivate
        self.model = AutoModel.from_pretrained(model_name)

        # freeze parameters in BERT model
        if freeze_params:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)

        print("Model " + self.tag + " initiated")

    def label_decide(self,output):
        return torch.argmax(output, dim=1) 

    def forward(self, ids_bert, attention_mask, token_type_ids, batch_ids): 
        # get embeddings from language model
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
        # aggregate over whole bag
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
        
        self.log("validation/loss", loss, on_epoch=True) #on_step=False
        self.log("validation/accuracy", self.valid_acc(pred, lbls.view(-1).int()), on_epoch=True) #on_step=False

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
        
        self.log("test/loss", loss, on_epoch=True) #on_step=False
        self.log("test/accuracy", self.test_acc(pred, lbls.view(-1).int()), on_epoch=True) #on_step=False

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

def aggregation_max(x, dim):
  return torch.max(x, dim)[0]


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from datasets.fingerprint import Hasher
from transformers import AutoModel, AutoTokenizer
import datasets
import wandb

from model_bert_baseline import *

approach = "_e2e"
model_name = "bert_baseline"
dataset_name = "20k_similar"

#EXPERIMENT_NAME = approach + "_" + model_name + "_" + dataset_name
class MultiSourceDataModule(pl.LightningDataModule):
    """
    This class was taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    def __init__(
        self,
        dataset_path: str = None,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        include_title: bool = False,
        batch_size: int = 1,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def _preprocess_example(example, tokenizer, max_length):
        preprocessed_example = {"label": torch.tensor([int(example["label"])])}

        preprocessed_example = preprocessed_example | tokenizer(
            " ".join(example["sentences"]),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            #return_tensor = "pt"
        )

        return preprocessed_example
    
    def load_my_dataset(self,dataset_dirr):
        dataset = datasets.load_dataset(
                "json",  # Using the datasets' loading scripts for jsonl files
                data_dir=dataset_dirr,
                data_files={
                    "train": os.path.join(dataset_dirr, "train.jsonl"),
                    "validation": os.path.join(dataset_dirr, "validation.jsonl"),
                    "test": os.path.join(dataset_dirr, "test.jsonl"),
                },
            )
        return dataset    
    
    def prepare_data(self):
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenized_dataset = self.load_my_dataset(
            self.dataset_path
        )

        # Preprocess the individual splits and assign custom fingerprints
        for split_name, split_data in self.tokenized_dataset.items():

            new_fingerprint = Hasher.hash(
                split_data._fingerprint
                + self.model_name
                + str(self.max_length)
                + Hasher.hash(self._preprocess_example)
            )

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, tokenizer, self.max_length),
                new_fingerprint=new_fingerprint,
                remove_columns=["ids", "sentences"],
            )

        # Set the format of the dataset to PyTorch
        self.tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "label"],
        )


    def train_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["test"],
            batch_size=self.batch_size,
        )
    
def launch_experiment(
        EXPERIMENT_NAME,
        DATASET, 
        name = "NOT FILLED IN",
        learning_rate = 0.0001,
        seed = 420,
        n_epochs = 20,
        batch_size = 40):
    
    print("****** " + EXPERIMENT_NAME + " ||| " + DATASET + " ******")
    pl.seed_everything(seed)
    
    #prepare data
    myDataModule = MultiSourceDataModule(dataset_path=DATASET, batch_size=batch_size)
    myDataModule.prepare_data()
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="validation/accuracy", mode="max") 

    # setup model
    model = BaselineBERT(
        model_name=myDataModule.model_name,
        weight_decay=0.001,
        num_labels=2,
        learning_rate=learning_rate)

    # define logger
    wandb_logger = WandbLogger(project=EXPERIMENT_NAME, log_model=True)
    wandb_logger.experiment.config["batch_size"] = batch_size

    # setup training
    trainer = Trainer(
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=n_epochs,
        val_check_interval=0.25,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        enable_progress_bar=False,
    )

    #setup logging
    wandb_logger.watch(model, log='all')
    wandb_logger.experiment.config["model_type"] = model.tag
    wandb_logger.experiment.config["dataset_destination"] = DATASET
    
    trainer.fit(model, myDataModule)
    print("TRAINING DONE")
    trainer.test(model, ckpt_path="best", dataloaders=myDataModule)
    print("TESTING DONE")
    wandb.finish()

launch_experiment(
    EXPERIMENT_NAME = "multisource_40k_similar",
    DATASET = '/home/jerabvo1/_data/multisource-40k-similar/',
    learning_rate = 2e-5,
    n_epochs=10,
    batch_size=40)
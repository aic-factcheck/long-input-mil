
import os
import sqlite3
import torch
import pytorch_lightning as pl
import pandas as pd
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer
import datasets
from datasets import Dataset, DatasetDict, load_from_disk
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track, Progress
from typing import Tuple
import wandb


from model_baseline import *

EXPERIMENT_NAME = "HYPERPARTISAN-THESIS-BASELINES"
def evaluate_model(
    model: torch.nn.Module, dataloader: DataLoader, display_progress=True
) -> Tuple[float, float]:
    """
    This function was taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    assert hasattr(
        model, "_common_step"
    ), "The model does not have the common step function"

    model.eval()
    total_loss, total_accuracy = 0, 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    with torch.no_grad(), Progress() as progress:
        if display_progress:
            task = progress.add_task("Evaluating the model...", total=len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            batch = {
                k: v.to(dev) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_loss, batch_accuracy, batch_predictions = model._common_step(
                batch, batch_idx, phase="train"
            )
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy

            if display_progress:
                progress.update(task, advance=1)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy


class HyperpartisanDataModule(LightningDataModule):
    """
    This class was taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    def __init__(
        self,
        dataset_path: str = None,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: int = 1,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    def prepare_data(self):
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenized_dataset = load_from_disk(self.dataset_path)

        self.tokenized_dataset = self.tokenized_dataset.map(
            lambda example: tokenizer(
                example["text"],
                max_length=512,
                padding="max_length",#takes only tokens limited to max length, 
                truncation=True,
            )
            | {"label": [example["label"]]}
        )

        # Compute the class weights for usage in training loss
        values, counts = torch.unique(
            torch.tensor(self.tokenized_dataset["train"]["label"]), return_counts=True
        )
        self.train_set_class_weights = (1 / len(values)) / counts

        # Set the format of the dataset to PyTorch
        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        tensor_columns = [ x for x in tensor_columns if x in self.tokenized_dataset["train"].column_names ]

        self.tokenized_dataset.set_format(type="torch", columns=tensor_columns)

    def get_train_class_weights(self):
        return self.train_set_class_weights

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
        model_name,
        batch_size,
        dataset_path,
        lr,
        epochs,
        seed = 420,
        
                      ):
    pl.seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    wandb_logger = WandbLogger(
            project=EXPERIMENT_NAME,
            name=f"hyperpartisan_{model_name}",
            log_model=True,
        )
    wandb_logger.experiment.config["batch_size"] = batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/f1_score", mode="max"),
    ]
    callbacks.append(RichProgressBar())

    # prepare data
    hyperpartisan = HyperpartisanDataModule(
        dataset_path=dataset_path,
        model_name=model_name,
        batch_size=batch_size,
    )
    hyperpartisan.prepare_data()

    # define model
    model = BaselineBERT(
        num_labels=2,
        learning_rate=lr,
        model_name=model_name,
        f1_log=True,
    )

    model.set_train_class_weights(
        hyperpartisan.get_train_class_weights().to(device)
    )

    # define validation metrics
    validation_metrics = evaluate_model(model, hyperpartisan.val_dataloader())

    # setup logging
    wandb_logger.experiment.log({
            "validation/loss": validation_metrics[0],
            "validation/accuracy": validation_metrics[1],
            "trainer/global_step": 0,
        })

    # setup training
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=0.25,
        enable_progress_bar=False,
        accumulate_grad_batches = 8, 
    )

    trainer.fit(model, hyperpartisan)
    wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=hyperpartisan)
    wandb.finish()

launch_experiment(
    model_name="allenai/longformer-base-4096",
    batch_size=4,
    dataset_path="/home/jerabvo1/_data/hyperpartisan",
    lr=0.00002,
    epochs=20,
    )

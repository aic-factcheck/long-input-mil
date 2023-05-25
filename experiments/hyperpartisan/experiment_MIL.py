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
from datasets import Dataset, DatasetDict, load_from_disk, set_caching_enabled
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track, Progress
from typing import Tuple
import wandb

set_caching_enabled(False)


from model_MIL import *
from pma import *

EXPERIMENT_NAME = "HYPERPARTISAN-THESIS-MIL"
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
            print("function called by evaluator")
            batch_loss, batch_accuracy, batch_predictions = model._common_step(
                batch, batch_idx, phase="train"
            )
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy

            if display_progress:
                progress.update(task, advance=1)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    print("sinished evaluating model")
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
    
    def collate_fn(self, batch):
        collated_batch = {
            "label": torch.concat([x["label"] for x in batch]).view(-1, 1),
            "input_ids": torch.concat([torch.vstack(x["input_ids"]) for x in batch]),
            "attention_mask": torch.concat(
                [torch.vstack(x["attention_mask"]) for x in batch]
            ),
            "batch_idx": torch.concat(
                [
                    torch.tensor([idx] * len(batch[idx]["input_ids"]))
                    for idx in range(len(batch))
                ]
            ),
        }

        if "token_type_ids" in batch[0].keys():
            collated_batch["token_type_ids"] = torch.concat(
                [torch.vstack(x["token_type_ids"]) for x in batch]
            )

        return collated_batch


    def prepare_data(self):
        # Load the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenized_dataset = load_from_disk(self.dataset_path)

        self.tokenized_dataset = self.tokenized_dataset.map(
            lambda example: tokenizer.encode_plus(
                example["text"],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens = True, 
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
            collate_fn = self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=self.batch_size,
            collate_fn = self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["test"],
            batch_size=self.batch_size,
            collate_fn = self.collate_fn,
        )
def launch_experiment(
        model_name,
        experiment_name,
        batch_size,
        dataset_path,
        lr,
        epochs,
        use_pma,
        use_max,
        seed = 420,
        input_size = 768,
        num_classes = 2,
    ):
    pl.seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    wandb_logger = WandbLogger(
            project=EXPERIMENT_NAME,
            name=experiment_name,
            log_model=True,
        )
    wandb_logger.experiment.config["batch_size"] = batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/f1_score.micro", mode="max"),
    ]
    callbacks.append(RichProgressBar())

    # prepare data
    hyperpartisan = HyperpartisanDataModule(
        dataset_path=dataset_path,
        model_name=model_name,
        batch_size=batch_size,
    )
    hyperpartisan.prepare_data()

    ##### PREPARE MODEL #####

    # preprocessing layer
    prepNN = torch.nn.Sequential( #768 -> 128, jedna vrstva
            torch.nn.Linear(input_size, 1268), #hardcoded number 768 because that is the size of embeddings
            torch.nn.ReLU(),
            torch.nn.Linear(1268,634),
            torch.nn.ReLU(),
            torch.nn.Linear(634,256),
            torch.nn.ReLU(),
    )

    # postprocessing layer
    afterNN = torch.nn.Sequential( # snizit hloubku, 2 vrstvy, moc vrstev mozna vanishing gradients, layer normalization, pojmenovat vrstvy pro lepsi prehlednost, kouknout na jejich gradienty
        torch.nn.Linear(256,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, num_classes), # changed from torch.nn.Linear(32, 2)
    )

    # select aggregation function
    if use_pma:
        agg_func = AttentionPool(256,256)
    elif use_max:
        agg_func = aggregation_max
    else:
        agg_func = torch.mean

    # define model
    model = BERTPlusMILModel(
        model_name=model_name,
        afterNN=afterNN,
        aggregation_func=agg_func,
        num_labels=2,
        out_features=2,
        learning_rate=lr,
        prepNN=prepNN,
        freeze_params=False,
        weight_decay=0.01,
    )

    # initiate class weights
    model.set_train_class_weights(
        hyperpartisan.get_train_class_weights().to(device)
    )

    ##### PREPARE TRAINER AND STUFF  ######
    print("preparin model for evaluation")

    # setup training
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=0.25,
        accumulate_grad_batches = 8, 
        enable_progress_bar=False,
        precision=16, #activates AMP,
        #deterministic = True,
    )

    trainer.fit(model, hyperpartisan)

    wandb_logger.experiment.config["epochs"] = trainer.current_epoch

    trainer.test(ckpt_path="best", dataloaders=hyperpartisan)
    wandb.finish()

# functions for extraction of parameters from bash script
def active_pma(args):
    if args.pma == 1:
        return True
    return False

def active_max(args):
    if args.max == 1:
        return True
    return False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--pma", type=int)
    parser.add_argument("--max", type=int)
    args = parser.parse_args()
    launch_experiment(
        model_name=args.model,
        experiment_name = args.experiment,
        batch_size=4,
        dataset_path="/home/jerabvo1/_data/hyperpartisan",
        lr=0.00002,
        epochs=20,
        use_pma = active_pma(args),
        use_max = active_max(args),
    )
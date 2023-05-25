import os
import sqlite3
import torch
import pytorch_lightning as pl
import pandas as pd
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
import datasets
from datasets import Dataset, DatasetDict, load_from_disk
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader
from rich.progress import track
from typing import Tuple
import wandb

from ctk import *
from typing import Tuple

from model_e2e_MIL import *
from pma import *

EXPERIMENT_NAME = "CTKNEWS-THESIS-MIL-EVIDENCE"

ARTICLES_DB_PATH = "/mnt/data/ctknews/factcheck/par5/interim/ctk_filtered.db"
RETRIEVAL_RESULTS_BASE_PATH = "/mnt/data/ctknews/factcheck/dataset/splits_concise_ctkId_s0_si0_t0.095_v0.12_source_77/models/anserini/predictions"

RETRIEVAL_RESULTS_PATHS = {
    "train": os.path.join(RETRIEVAL_RESULTS_BASE_PATH, "train_anserini_k500_0.6_0.5.jsonl"),
    "validation": os.path.join(RETRIEVAL_RESULTS_BASE_PATH, "validation_anserini_k500_0.6_0.5.jsonl"),
    "test": os.path.join(RETRIEVAL_RESULTS_BASE_PATH, "test_anserini_k500_0.6_0.5.jsonl")
}


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

def load_jsonl(filepath: str, encoding="utf-8", ensure_ascii=False):
    output = []

    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            record = json.loads(line)
            output.append(record)

    return output


def get_evidence(x, k=None):
    k = min(k, len(x["predicted_pages"])) if k else len(x["predicted_pages"])
    evidence_ids = x["predicted_pages"][:k]

    with sqlite3.connect(ARTICLES_DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM documents WHERE id IN ({})".format(
            "'" + "', '".join(evidence_ids) + "'"
        ), con).set_index("id")

        return df.loc[evidence_ids]["text"].to_list()


def convert_list_to_dict(dataset: list, ignore_fields: list = None):
    keys = [ key for key in dataset[0].keys() if key not in ignore_fields ]
    output = { key: [] for key in keys }
    for x in dataset:
        for key in keys:
            output[key].append(x[key])

    return output


def load_ctknews(paths: dict, k: int = 10):
    LABELS = { "REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2 }

    splits = {
        name: load_jsonl(path)
        for name, path in paths.items()
    }

    for split_name in splits.keys():
        for example_idx in range(len(splits[split_name])):
            splits[split_name][example_idx]["label"] = LABELS[
                splits[split_name][example_idx]["label"]
            ]
            splits[split_name][example_idx]["evidence"] = get_evidence(
                splits[split_name][example_idx], k=k
            )

    splits = {
        k: convert_list_to_dict(v, ignore_fields=["id", "verifiable", "predicted_pages"])
        for k, v in splits.items()
    }

    splits = {
        k: datasets.Dataset.from_dict(v)
        for k, v in splits.items()
    }

    return datasets.DatasetDict(splits)



class CTKNewsDataModule(pl.LightningDataModule):
    """
    This class was taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
    """
    def __init__(
        self,
        train_split_path: str,
        validation_split_path: str,
        test_split_path: str,
        articles_db_path: str,
        evidence_count: int = 10,
        model_name: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: int = 2
    ):
        super().__init__()

        self.train_split_path = train_split_path
        self.validation_split_path = validation_split_path
        self.test_split_path = test_split_path
        self.articles_db_path = articles_db_path
        self.evidence_count = evidence_count
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_class_weights = None

    @staticmethod
    def _preprocess_example(example: dict, tokenizer: PreTrainedTokenizer, max_length: int):
        preprocessed_example = { "label": torch.tensor([example["label"]]) }

        evidence_sequences = example["evidence"]

        preprocessed_example = preprocessed_example | tokenizer(
            text=[example["claim"]] * len(evidence_sequences),
            text_pair=evidence_sequences,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example

    def collate_fn(self, batch):
        collated_batch = {
            "label": torch.concat([ x["label"] for x in batch ]).view(-1, 1),
            "input_ids": torch.concat([torch.vstack(x["input_ids"]) for x in batch]), #TypeError: expected Tensor as element 0 in argument 0, but got list
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenized_dataset = load_ctknews({
            "train": self.train_split_path,
            "validation": self.validation_split_path,
            "test": self.test_split_path
        }, k=self.evidence_count)

        # Preprocess the individual splits and assign custom fingerprints
        for split_name, split_data in self.tokenized_dataset.items():

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, self.tokenizer, self.max_length),
            )

        # Compute the class weights for usage in training loss
        values, counts = torch.unique(
            torch.tensor(self.tokenized_dataset["train"]["label"]), return_counts=True
        )
        self.train_class_weights = (1 / len(values)) / counts

        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        tensor_columns = list(
            filter(
                lambda x: x in self.tokenized_dataset["train"].column_names,
                tensor_columns,
            ),
        )

        # Set the format of the dataset to PyTorch
        self.tokenized_dataset.set_format(type="torch", columns=tensor_columns)


    def get_train_class_weights(self):
        return self.train_class_weights

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["test"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

    
def launch_experiment(
        model_name,
        evidence_count,
        EXPERIMENT_NAME,
        DATASET,
        use_pma = False,
        use_max = False,
        name = "NOT FILLED IN",
        learning_rate = 0.0001,
        seed = 420,
        n_epochs = 10,
        batch_size = 2,
        input_size = 768,
        num_classes = 3,
):
    print("****** " + EXPERIMENT_NAME + " ||| " + DATASET + " ******")
    print(name)
    pl.seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    wandb_logger = WandbLogger(
        project=EXPERIMENT_NAME,
        name=f"ev_ctknews_{model_name}_{evidence_count}-evidence",
        log_model=True,
    )
    wandb_logger.experiment.config["batch_size"] = batch_size

    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor="validation/f1_score.micro", mode="max"),
    ]
    callbacks.append(RichProgressBar())

    # prepare data
    ctknews = CTKNewsDataModule(
        train_split_path=RETRIEVAL_RESULTS_PATHS["train"],
        validation_split_path=RETRIEVAL_RESULTS_PATHS["validation"],
        test_split_path=RETRIEVAL_RESULTS_PATHS["test"],
        model_name=model_name,
        articles_db_path=ARTICLES_DB_PATH,
        evidence_count=evidence_count,
        batch_size=batch_size
    )
    ctknews.prepare_data()

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

    # define aggregation function
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
        num_labels=3,
        out_features=3,
        learning_rate=learning_rate,
        prepNN=prepNN,
        freeze_params=False,
        weight_decay=0.01,
        
        )
    
    model.set_train_class_weights(
        ctknews.get_train_class_weights().to(device) # ??
    )

    ##### PREPARE TRAINER AND STUFF  ######

    # setup validation metrics and log their initial values
    validation_metrics = evaluate_model(model, ctknews.val_dataloader())
    wandb_logger.experiment.log({
        "validation/loss": validation_metrics[0],
        "validation/accuracy": validation_metrics[1],
        "validation/f1_score": -1,
        "trainer/global_step": 0,
    })

    # setup training
    trainer = Trainer(
        accelerator="auto",
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=n_epochs,
        val_check_interval=0.25,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        accumulate_grad_batches = 8, 
        precision=16 #activates AMP
    )

    # add more logging
    wandb_logger.watch(model, log='all')
    wandb_logger.experiment.config["classifer_type"] = model.tag
    wandb_logger.experiment.config["dataset_destination"] = DATASET
    wandb_logger.experiment.config["agg"] = "agg-PMA"

    trainer.fit(model, ctknews)
    print("TRAINING DONE")
    trainer.test(model, ckpt_path="best", dataloaders=ctknews)
    print("TESTING DONE")
    wandb.finish()


def convert_args(args):
    model = args.model
    evidence = args.evidence
    return model, evidence
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
    parser.add_argument("--evidence", type=int)
    parser.add_argument("--pma", type=int)
    parser.add_argument("--max", type=int)
    args = parser.parse_args()
    model_name, evidence_count = convert_args(args)
    launch_experiment(
        EXPERIMENT_NAME = EXPERIMENT_NAME,
        model_name=model_name,
        DATASET = "CTK_facts",
        use_pma = active_pma(args),
        use_max = active_max(args),
        learning_rate = 2e-5,
        name = "E2E ctk facts set with PMA with grad accumulation, 10 grad_acc, evidence 15",
        n_epochs=20,
        batch_size=4,
        evidence_count=evidence_count)




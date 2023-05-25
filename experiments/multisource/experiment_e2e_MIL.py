
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

from model_e2e_MIL import *
from pma import *




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
            example["sentences"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        return preprocessed_example
    
    def load_my_dataset(self,dataset_dirr):
        # COULD BE HELPER FUNCTION
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

    def collate_fn(self, batch):
        for x in batch:
            print(len(x["input_ids"]), type(x["input_ids"]))
            print(x["label"])
            for y in x["input_ids"]:
                print(type(y), y.shape)
                
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

            columns_to_remove = ["ids", "sentences"]
            columns_to_remove = list(
                filter(
                    lambda x: x in self.tokenized_dataset[split_name].column_names,
                    columns_to_remove,
                )
            )

            self.tokenized_dataset[split_name] = self.tokenized_dataset[split_name].map(
                lambda x: self._preprocess_example(x, tokenizer, self.max_length),
                new_fingerprint=new_fingerprint,
                remove_columns=columns_to_remove,
            )

        tensor_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        tensor_columns = list(
            filter(
                lambda x: x in self.tokenized_dataset["train"].column_names,
                tensor_columns,
            ),
        )

        # Set the format of the dataset to PyTorch
        self.tokenized_dataset.set_format(type="torch", columns=tensor_columns)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
        )
    
def launch_experiment(
        EXPERIMENT_NAME,
        DATASET,
        use_pma = False,
        use_max = False,
        name = "NOT FILLED IN",
        learning_rate = 0.0001,
        seed = 420,
        n_epochs = 20,
        batch_size = 2,
        grad_acc = 10,
        num_classes = 2,
        input_size = 768,
):
    print("****** " + EXPERIMENT_NAME + " ||| " + DATASET + " ******")
    print(name)
    pl.seed_everything(seed)

    #preprocessing layer
    prepNN = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1268), 
            torch.nn.ReLU(),
            torch.nn.Linear(1268,634),
            torch.nn.ReLU(),
            torch.nn.Linear(634,256),
            torch.nn.ReLU(),
    )

    #postprocessing layer
    afterNN = torch.nn.Sequential( 
        torch.nn.Linear(256,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, num_classes), 
    )

    # define aggregation function
    if use_pma:
        agg_func = AttentionPool(dim_in=256, dim_out=256)
    elif use_max:
        agg_func = aggregation_max
    else:
        agg_func = torch.mean

    # prepare data
    myDataModule = MultiSourceDataModule(dataset_path=DATASET, batch_size=batch_size)
    myDataModule.prepare_data()
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="validation/accuracy", mode="max")

    # define model 
    model = BERTPlusMILModel(
        afterNN=afterNN,
        aggregation_func=agg_func,
        num_labels=2,
        learning_rate=learning_rate,
        prepNN=prepNN,
        freeze_params=False)

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
        accumulate_grad_batches = grad_acc, 
        precision=16 #activates AMP
    )

    # setup logging
    wandb_logger.watch(model, log='all')
    wandb_logger.experiment.config["model_type"] = model.tag
    wandb_logger.experiment.config["dataset_destination"] = DATASET
    wandb_logger.experiment.config["grad_acc"] = grad_acc

    trainer.fit(model, myDataModule)
    print("TRAINING DONE")
    trainer.test(model, ckpt_path="best", dataloaders=myDataModule)
    print("TESTING DONE")
    wandb.finish()

launch_experiment(
    EXPERIMENT_NAME = "multisource_40k_similar",
    DATASET = '/home/jerabvo1/_data/multisource-40k-similar/',
    use_pma = False,
    use_max = True,
    name = "E2E 40k set with MAX with grad accumulation, grad_acc 10, batch size 40",
    learning_rate = 2e-5,
    n_epochs=20,
    batch_size=4)

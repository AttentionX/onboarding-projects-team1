"""
Train pegasus by fine-tuning it on the samsum dataset.
All the weights are optimized, so catastrophic forgetting is possible.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Union, List
import torch
import wandb
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger  # noqa
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nn_pegasus_soft import PegasusSoftForConditionalGeneration
from torch.utils.data import TensorDataset, DataLoader
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("approach", type=str, choices=["ft", "pt"], help="either ft(fine-tuning) or pt(prompt-tuning)")
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=5)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=-1)
parser.add_argument("--shuffle", type=int, default=1, choices=[0, 1])
parser.add_argument("--fast_dev_run", type=int, default=1, choices=[0, 1])
parser.add_argument("--log_model", type=int, default=1, choices=[0, 1])
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--limit_val_batches", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=82)
parsed_args = parser.parse_args()
with open(Path(__file__).resolve().parent / "config.yaml", 'r') as fh:
    config = yaml.safe_load(fh)
config.update(vars(parsed_args))
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class PegasusModule(pl.LightningModule):
    """
    Just for training pegasus.  Nothing fancy.
    """
    def __init__(self, pegasus: Union[PegasusForConditionalGeneration, PegasusSoftForConditionalGeneration],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pegasus = pegasus

    def training_step(self, batch: List[torch.LongTensor], *args, **kwargs) -> dict:
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        output = self.pegasus.forward(input_ids, attention_mask, labels=labels)
        loss = output['loss']
        self.log("Train/loss", loss)
        return {
            'loss': output['loss']  # the training loss
        }

    def validation_step(self, batch: List[torch.LongTensor], *args, **kwargs) -> dict:
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        output = self.pegasus.forward(input_ids, attention_mask, labels=labels)
        loss = output['loss']
        self.log("Validation/loss", loss)
        return {
            'loss': output['loss']  # the validation loss
        }

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), **wandb.config["optimizer"])
        return {
            "optimizer": optimizer
        }

def main():

    with wandb.init(project="onboarding-projects-team1", job_type=os.path.basename(__file__), config=config) as run:
        # --- load pre-trained tokenizer & pegasus --- #
        name = "google/pegasus-cnn_dailymail"
        if config['approach'] == "ft":
            pegasus = PegasusForConditionalGeneration.from_pretrained(name)
        elif config['approach'] == "pt":
            pegasus = PegasusSoftForConditionalGeneration.from_pretrained(name)
        else:
            ValueError(f"Unknown approach: {config['approach']}")
        tokenizer = PegasusTokenizer.from_pretrained(name)
        # --- tokenize texts --- #
        samsum_dataset = load_dataset('samsum')
        train_encodings = tokenizer([example['dialogue'] for example in samsum_dataset['train']],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=config['max_length'],
                                    return_tensors="pt")
        train_labels = tokenizer([example['summary'] for example in samsum_dataset['train']],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=config['max_length'],
                                    return_tensors="pt")['input_ids']
        val_encodings = tokenizer([example['dialogue'] for example in samsum_dataset['validation']],
                                  add_special_tokens=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=config['max_length'],
                                  return_tensors="pt")
        val_labels = tokenizer([example['summary'] for example in samsum_dataset['validation']],
                               add_special_tokens=True,
                               padding='max_length',
                               truncation=True,
                               max_length=config['max_length'],
                               return_tensors="pt")['input_ids']
        # --- build datasets to instantiate dataloaders --- #
        train_dataset = TensorDataset(train_encodings['input_ids'],
                                      train_encodings['attention_mask'],
                                      train_labels)
        val_dataset = TensorDataset(val_encodings['input_ids'],
                                    val_encodings['attention_mask'],
                                    val_labels)
        g = torch.Generator()
        g.manual_seed(parsed_args.seed)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=parsed_args.batch_size,
                                      shuffle=parsed_args.shuffle,
                                      num_workers=parsed_args.num_workers,
                                      generator=g)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=parsed_args.batch_size,
                                    num_workers=parsed_args.num_workers,
                                    generator=g)
        # --- train pegasus --- #
        trainer = pl.Trainer(
            logger=WandbLogger(log_model=parsed_args.log_model),
            max_epochs=parsed_args.max_epochs,
            accelerator=parsed_args.accelerator,
            devices=parsed_args.devices,
            fast_dev_run=parsed_args.fast_dev_run,
            limit_train_batches=parsed_args.limit_train_batches,
            limit_val_batches=parsed_args.limit_val_batches
        )
        trainer.fit(PegasusModule(pegasus),
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)
        # --- persist the final artifacts to wandb only if the training is properly done --- #
        if not parsed_args.fast_dev_run and not trainer.interrupted:
            artifact = wandb.Artifact(f"pegasus", type="model")
            save_dir = Path("out") / str(datetime.now())
            os.makedirs(save_dir)
            tokenizer.save_pretrained(save_dir / "tokenizer")
            pegasus.save_pretrained(save_dir / "pegasus")
            artifact.add_dir(str(save_dir))
            run.log_artifact(artifact)




if __name__ == '__main__':
    main()
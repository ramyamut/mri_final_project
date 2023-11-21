import torch
import os
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src import metrics

class ReconLightningModule(pl.LightningModule):
    """
    Lightning Module class for training reconstruction model
    """
    def __init__(self, config, datasets, model, log_dir):
        super().__init__()
        self.config = config
        self.datasets = datasets
        self.model = model
        self.log_dir = log_dir
        self.parse_config()
        self.collate_fn = None
        self.outputs = {
            "train": [],
            "val": [],
            "test": []
        }
        self.metrics = {}
    
    def parse_config(self):
        self.batch_size = int(self.config["batch_size"])
        self.lr = float(self.config["lr"])
        self.weight_decay = float(self.config["weight_decay"])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def step(self, batch, stage):
        kspace = batch['kspace'] # [B, 2, 1, H, W]
        recon = batch['recon'] # [B, 2, 1, H, W]
        preds = self.model(kspace, recon) # [B, 2, 1, H, W]
        loss = metrics.mse(recon, preds)
        output_dict = {
            'loss': loss.mean()
        }
        self.outputs[stage].append(output_dict)
        return output_dict
    
    def training_step(self, batch, _):
        return self.step(batch, "train")
    
    def validation_step(self, batch, _):
        return self.step(batch, "val")
    
    def test_step(self, batch, _):
        return self.step(batch, "test")
    
    def predict_step(self, batch, _):
        return self.step(batch)
    
    def epoch_end(self, stage):
        outputs = self.outputs[stage]
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_metric(f"{stage}/loss", loss)
        self.save_metrics()
        self.outputs[stage] = []
    
    def on_train_epoch_end(self):
        self.epoch_end("train")
    
    def on_validation_epoch_end(self):
        self.epoch_end("val")
    
    def on_test_epoch_end(self):
        self.epoch_end("test")
    
    def log_metric(self, metric_name, metric_value):
        self.log(metric_name, metric_value, on_epoch=True, prog_bar=True)
        if metric_name in self.metrics:
            self.metrics[metric_name].append(metric_value.item())
        else:
            self.metrics[metric_name] = [metric_value.item()]
    
    def save_metrics(self):
        filename = os.path.join(self.log_dir, "metrics.json")
        with open(filename, "w") as f:
            json.dump(self.metrics, f)
    
    def train_dataloader(self):
        train_dataset = self.datasets["train"]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        eval_dataset = self.datasets["val"]
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        return eval_loader
    
    def test_dataloader(self):
        test_dataset = self.datasets["test"]
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader

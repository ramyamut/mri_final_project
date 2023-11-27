# python imports
import os
import numpy as np
import pytorch_lightning as pl

# project imports
from src import networks, dataset, lightning

def training(train_kspace_dir,
             val_kspace_dir,
             model_dir,
             subsample_method=None,
             subsample_factor=1,
             batchsize=1,
             hidden_channels=32,
             num_layers=5,
             layer_type='interleaved',
             lr=1e-4,
             weight_decay=0,
             n_epochs=500,
             checkpoint=None):
    
    train_dataset = dataset.ReconDataset(
        kspace_dir=train_kspace_dir,
        subsample_method=subsample_method,
        subsample_factor=subsample_factor
    )
    val_dataset = dataset.ReconDataset(
        kspace_dir=val_kspace_dir,
        subsample_method=subsample_method,
        subsample_factor=subsample_factor,
        eval_mode=True
    )
    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    # create network and lightning module
    channels = [hidden_channels] * (num_layers - 1)
    net = networks.Net(
        channels=channels,
        layer_type=layer_type
    )
    module_config = {
        'batch_size': batchsize,
        'lr': lr,
        'weight_decay': weight_decay
    }
    module = lightning.ReconLightningModule(
        config=module_config,
        datasets=datasets,
        model=net,
        log_dir=model_dir
    )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val/loss",
        mode="min",
        dirpath=model_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    # build trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=n_epochs,
        default_root_dir=model_dir,
        callbacks=[checkpoint_callback]
    )

    # train model
    trainer.fit(module, ckpt_path=checkpoint)
    module.save_metrics()

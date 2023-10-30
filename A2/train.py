import argparse

import torch
from dataset import LitDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from model import LitModule
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("high")

def train(cfg: OmegaConf, debug=False):
    cfg.data.dataset.triplet = cfg.model.triplet.use
    cfg.train.batch_size = cfg.data.dataloader.batch_size
    datamodule = LitDataModule(cfg.data)
    datamodule.setup()
    cfg.model.num_classes = datamodule.get_num_classes()

    model = LitModule(cfg.model, cfg.train)

    loss_model_checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        filename=f"{cfg.model.encoder.name}_{cfg.model.num_classes}_classes_{cfg.train.precision}_f_best_loss_" + "{val_loss:.4f}",
        verbose="True",
        auto_insert_metric_name=False,
    )

    acc_model_checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_acc",
        mode="max",
        filename=f"{cfg.model.encoder.name}_{cfg.model.num_classes}_classes_{cfg.train.precision}_f_best_acc_" + "{val_acc:.4f}",
        verbose="True",
        auto_insert_metric_name=False,
    )

    callbacks = [loss_model_checkpoint, acc_model_checkpoint]
    if cfg.train.swa.use:
        swa = StochasticWeightAveraging(swa_lrs=cfg.train.swa.swa_lr, swa_epoch_start=1, avg_fn=None)
        callbacks.append(swa)
    trainer = Trainer(
        fast_dev_run=debug,
        callbacks=callbacks,
        benchmark=False,  # ENSURE REPRODUCIBILITY
        deterministic=True,  # ENSURE REPRODUCIBILITY
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy="ddp" if cfg.train.ddp else "auto",
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        logger=WandbLogger(project=cfg.project, name=f"{cfg.model.encoder.name}-{cfg.train.precision}"),
    )
    if cfg.train.optimizer.auto_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=datamodule)
        model.hparams.lr = lr_finder.suggestion()
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        metavar="",
        help="configuration file",
        default="config/default.yaml",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)

    train(cfg, debug=args.debug)

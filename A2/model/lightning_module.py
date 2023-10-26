from typing import Dict, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import OmegaConf
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchmetrics.classification import (Accuracy, F1Score, Precision, Recall,
                                         Specificity)

from .triplet_model import TripletModel


class LitModule(LightningModule):
    def __init__(self, model_cfg: OmegaConf, train_cfg: Optional[OmegaConf] = None) -> None:
        super(LitModule, self).__init__()
        self.opt_cfg = train_cfg
        self.batch_size = train_cfg.batch_size
        self.triplet = model_cfg.triplet.use
        self.model = TripletModel(
            num_classes=model_cfg.num_classes,
            encoder_name=model_cfg.encoder.name,
            pretrained=model_cfg.encoder.pretrained,
            features_only=True if self.triplet else False,
            skip_connection=model_cfg.triplet.skip_connection,
            pe=model_cfg.triplet.positional_encoding,
            mha=model_cfg.triplet.multi_head_attention,
            dropout=model_cfg.encoder.drop_rate,
        )
        if model_cfg.encoder.freeze_encoder:
            for k, item in dict(self.model.encoder.named_children()).items():
                if k != "global_pool" or k != "fc":
                    item.requires_grad = False
        if self.triplet:
            distance_fn = nn.PairwiseDistance() if model_cfg.triplet.distance == "l2" else nn.CosineSimilarity()
            self.triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, swap=True)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=model_cfg.num_classes)
        self.precision_fn = Precision(task="multiclass", num_classes=model_cfg.num_classes)
        self.recall_fn = Recall(task="multiclass", num_classes=model_cfg.num_classes)
        self.specificity_fn = Specificity(task="multiclass", num_classes=model_cfg.num_classes)
        self.f1score_fn = F1Score(task="multiclass", num_classes=model_cfg.num_classes)
        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=["cfg"])
    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.triplet:
            outputs = self.model(X["image"], X["positive"], X["negative"])
        else:
            outputs = self.model(X["image"])
        return outputs
    def configure_optimizers(self):
        self.lr = self.opt_cfg.optimizer.lr
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.opt_cfg.optimizer.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.opt_cfg.scheduler.T_0,
            T_mult=self.opt_cfg.scheduler.T_mult,
            eta_min=self.opt_cfg.scheduler.eta_min,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch).transpose(2, 1).softmax(dim=-1)
    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        outputs = self(batch)
        if self.triplet:
            outputs, anchors, positives, negatives = outputs
            _triplet_loss = self.triplet_loss_fn(anchors, positives, negatives)
            self.log(f"{step}_triplet_loss", _triplet_loss, sync_dist=True, batch_size=self.batch_size)
            _ce_loss = self.ce_loss_fn(outputs, batch["label"])
            self.log(f"{step}_ce_loss", _ce_loss, sync_dist=True, batch_size=self.batch_size)
            loss = (_triplet_loss + _ce_loss) / 2
            self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True, batch_size=self.batch_size)
        else:
            loss = self.ce_loss_fn(outputs, batch["label"])
            self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True, batch_size=self.batch_size)
        with torch.no_grad():
            _acc = self.accuracy_fn(outputs, batch["label"])
            self.log(f"{step}_accuracy", _acc, sync_dist=True, batch_size=self.batch_size)
            _prec = self.precision_fn(outputs, batch["label"])
            self.log(f"{step}_precision", _prec, sync_dist=True, batch_size=self.batch_size)
            _recall = self.recall_fn(outputs, batch["label"])
            self.log(f"{step}_recall", _recall, sync_dist=True, batch_size=self.batch_size)
            _spec = self.specificity_fn(outputs, batch["label"])
            self.log(f"{step}_specificity", _spec, sync_dist=True, batch_size=self.batch_size)
            _f1 = self.f1score_fn(outputs, batch["label"])
            self.log(f"{step}_f1score", _f1, sync_dist=True, batch_size=self.batch_size)
        return loss
from typing import Optional

import timm
import torch
import torch.nn as nn
from einops import pack, rearrange, unpack

from .positional_encoding import PositionalEncoding1D


class TripletModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "resnet34",
        pretrained: Optional[bool] = None,
        features_only: Optional[bool] = None,
        skip_connection: Optional[bool] = None,
        pe: Optional[bool] = None,
        mha: Optional[bool] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ) -> None:
        super(TripletModel, self).__init__()
        self.encoder = timm.create_model(
            model_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=features_only,
            drop_rate=dropout,
        )

        self.features_only = features_only
        if features_only:
            self.skip_connection = skip_connection
            feature_info = self.encoder.feature_info
            last_dim = sum([item["num_chs"] for item in feature_info]) if skip_connection else list(feature_info)[-1]["num_chs"]
            self.pool_flat = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self.embedding = nn.Linear(last_dim, last_dim, bias=False)
            self.pe = PositionalEncoding1D(last_dim) if pe else nn.Identity()
            self.mha = nn.MultiheadAttention(embed_dim=last_dim, num_heads=8, bias=False, batch_first=True) if mha else None
            self.head = nn.Sequential(
                nn.Linear(last_dim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout if dropout else 0.0),
                nn.Linear(256, 64, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes, bias=False),
            )
    def _forward_base(self, X):
        return self.encoder(X)
    def _forward_triplet_infer(self, anchor):
        x = self.encoder(anchor) if self.skip_connection else self.encoder(anchor)[-1]
        if self.skip_connection:
            x = torch.concat([self.pool_flat(feats) for feats in x], dim=1)
        else:
            x = self.pool_flat(x)
        before = x
        # Embedding
        x = self.embedding(x)
        x = self.pe(x.unsqueeze(-1)).squeeze(-1)
        # Multi-head Attention
        if self.mha:
            x, _ = self.mha(x, x, x)
        after = x
        x = self.head(x)
        return x, before, after
    def _forward_triplet_train(self, anchor, positive, negative):
        x = [self.encoder(item) if self.skip_connection else self.encoder(item)[-1] for item in [anchor, positive, negative]]
        if self.skip_connection:
            x = [
                torch.concat(
                    [
                        self.pool_flat(rearrange(feats, "b h w c -> b c h w")) if feats.shape[1] == feats.shape[2] else self.pool_flat(feats)
                        for feats in feats_lst
                    ],
                    dim=1,
                )
                for feats_lst in x
            ]
        else:
            x = [self.pool_flat(rearrange(feats, "b h w c -> b c h w")) if feats.shape[1] == feats.shape[2] else self.pool_flat(feats) for feats in x]

        # Embedding
        x = [self.embedding(item) for item in x]
        x, x_ind = pack(x, "b * c")
        x = self.pe(x)
        x = unpack(x, x_ind, "b * c")
        # Multi-head Attention
        if self.mha:
            x = [self.mha(item, item, item)[0] for item in x]
        x = [self.head(item) for item in x]
        anchor, positive, negative = x
        return anchor, positive, negative
    def _forward_triplet(self, anchor, positive=None, negative=None):
        if positive == None and negative == None:
            return self._forward_triplet_infer(anchor)
        elif positive != None and negative != None:
            return self._forward_triplet_train(anchor, positive, negative)
        else:
            raise Exception("check inputs")
    def forward(
        self,
        X,
        positive=None,
        negative=None,
    ):
        return self._forward_triplet(X, positive, negative) if self.features_only else self._forward_base(X)

if __name__ == "__main__":
    _input0 = torch.randn(8, 3, 224, 224)
    _input1 = torch.randn(8, 3, 224, 224)
    _input2 = torch.randn(8, 3, 224, 224)
    model = TripletModel(10, features_only=False)
    output = model(_input0)
    print(output.shape)
    model = TripletModel(10, features_only=True, skip_connection=True, pe=True, mha=True, dropout=0.5)
    output = model(_input0, _input1, _input2)
    for i in output:
        print(i.shape)
    output = model(_input0)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)

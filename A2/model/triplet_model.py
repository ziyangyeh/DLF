from typing import Optional

import timm
import torch
import torch.nn as nn
from einops import pack, rearrange, unpack

class TripletModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "resnet34",
        pretrained: Optional[bool] = None,
        features_only: Optional[bool] = None,
        skip_connection: Optional[bool] = None,
        dropout: Optional[float] = 0.0,
        init: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(TripletModel, self).__init__()
        self.vit = True if encoder_name.startswith("vit") else False
        self.encoder = timm.create_model(
            model_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=features_only if not self.vit else None,
            drop_rate=dropout,
        )

        self.features_only = features_only
        if features_only:
            self.skip_connection = skip_connection
            if not self.vit:
                feature_info = self.encoder.feature_info
                last_dim = sum([item["num_chs"] for item in feature_info]) if skip_connection else list(feature_info)[-1]["num_chs"]
            else:
                last_dim = self.encoder.head.in_features
                self.encoder.head = nn.Identity()
            self.pool_flat = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()) if not self.vit else nn.Identity()
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
            if init:
                self.init_type = init
                self.head.apply(self.init_weights)
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            elif self.init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            elif self.init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
    def _forward_base(self, X):
        return self.encoder(X)
    def _forward_triplet_infer(self, anchor):
        x = self.encoder(anchor) if self.skip_connection else self.encoder(anchor)[-1]
        if self.skip_connection:
            x = torch.concat([self.pool_flat(feats) for feats in x], dim=1)
        else:
            x = self.pool_flat(x)
        x = self.head(x)
        return x, x, x
    def _forward_triplet_train(self, anchor, positive, negative):
        x = [self.encoder(item) if self.skip_connection else self.encoder(item)[-1] for item in [anchor, positive, negative]]
        if not self.vit:
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
                x = [
                    self.pool_flat(rearrange(feats, "b h w c -> b c h w")) if feats.shape[1] == feats.shape[2] else self.pool_flat(feats)
                    for feats in x
                ]
        else:
            x = [self.pool_flat(feats) for feats in x]
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

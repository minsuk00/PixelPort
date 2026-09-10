# Trimmed from anatomix (https://github.com/neel-dey/anatomix), MIT License,
# Copyright 2024 Neel Dey.

import torch
import torch.nn as nn


class Unet(nn.Module):
    """3D U-Net: `num_downs` encoder levels, channel width doubling from `ngf`."""

    def __init__(self, input_nc, output_nc, num_downs, ngf):
        super().__init__()

        def conv(cin, cout):
            return nn.Conv3d(cin, cout, 3, padding="same", padding_mode="reflect", bias=False)

        def block(cin, cout):
            return [conv(cin, cout), nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]

        model = block(input_nc, ngf)
        self.encoder_idx = []
        ch = ngf
        for i in range(num_downs):
            out = ch if i == 0 else ch * 2
            model += block(ch, out) + block(out, out)
            self.encoder_idx.append(len(model) - 1)
            model += [nn.MaxPool3d(2)]
            ch = out

        model += block(ch, ch * 2) + block(ch * 2, ch * 2)

        self.decoder_idx = []
        mult = 2**num_downs
        for _ in range(num_downs):
            self.decoder_idx.append(len(model))
            model += [nn.Upsample(scale_factor=2, mode="nearest")]
            out = ngf * (mult // 2)
            model += block(ngf * (mult + mult // 2), out) + block(out, out)
            mult //= 2

        model += [conv(ngf * mult, output_nc)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.decoder_idx:
                x = torch.cat((skips.pop(), x), dim=1)
            if i in self.encoder_idx:
                skips.append(x)
        return x
